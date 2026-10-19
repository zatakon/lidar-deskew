"""Reading ROS1 bags and MCAP files, writing MCAP files.

No ROS installation is needed: the lidar messages are decoded by the `rosbags` package, all
other messages are copied as they are (with their schema and encoding).
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class RawMessage:
    topic: str
    msgtype: str              # e.g. 'sensor_msgs/msg/PointCloud2'
    encoding: str             # 'ros1' or 'cdr'
    schema_name: str
    schema_encoding: str      # 'ros1msg' or 'ros2msg'
    schema_data: bytes
    log_time: int             # [ns]
    publish_time: int         # [ns]
    data: bytes


def _normalize_type(msgtype: str) -> str:
    """'sensor_msgs/PointCloud2' -> 'sensor_msgs/msg/PointCloud2'"""
    parts = msgtype.split('/')
    if len(parts) == 2:
        return f'{parts[0]}/msg/{parts[1]}'
    return msgtype


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class Recording:
    """Uniform access to the messages of a ROS1 bag or an MCAP file."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f'Recording not found: {self.filepath}')
        suffix = self.filepath.suffix.lower()
        if suffix == '.bag':
            self.kind = 'bag'
        elif suffix == '.mcap':
            self.kind = 'mcap'
        else:
            raise ValueError(f'Unsupported recording "{self.filepath}" - expected a .bag or .mcap file')
        self._typestores = {}

    @property
    def profile(self) -> str:
        if self.kind == 'bag':
            return 'ros1'
        from mcap.reader import make_reader
        with open(self.filepath, 'rb') as file:
            return make_reader(file).get_header().profile

    def topics(self) -> Dict[str, str]:
        """Topic name -> message type (e.g. 'sensor_msgs/msg/PointCloud2')."""
        if self.kind == 'bag':
            from rosbags.rosbag1 import Reader
            with Reader(self.filepath) as reader:
                return {connection.topic: connection.msgtype for connection in reader.connections}

        from mcap.reader import make_reader
        with open(self.filepath, 'rb') as file:
            summary = make_reader(file).get_summary()
            if summary is None:
                raise ValueError(f'"{self.filepath}" has no summary section (not finished writing?)')
            return {channel.topic: _normalize_type(summary.schemas[channel.schema_id].name)
                    for channel in summary.channels.values()}

    def find_topic(self, type_suffix: str, requested: Optional[str] = None) -> str:
        """The requested topic, or the only topic whose type ends with `type_suffix`."""
        topics = self.topics()
        if requested:
            if requested not in topics:
                raise ValueError(f'"{self.filepath}" has no topic "{requested}", it has: {sorted(topics)}')
            return requested
        matching = [topic for topic, msgtype in topics.items() if msgtype.endswith(type_suffix)]
        if len(matching) != 1:
            raise ValueError(f'"{self.filepath}": expected exactly one {type_suffix} topic, found {sorted(matching)}. '
                             f'Specify the topic to be used.')
        return matching[0]

    def messages(self, topics=None, begin: Optional[int] = None, end: Optional[int] = None) -> Iterator[RawMessage]:
        """
        The messages of the given topics (all if None) in the order of their log time, undecoded.

        @param begin, end: log time window [ns], both inclusive
        """
        if self.kind == 'bag':
            from rosbags.rosbag1 import Reader
            with Reader(self.filepath) as reader:
                connections = [connection for connection in reader.connections
                               if topics is None or connection.topic in topics]
                stop = None if end is None else end + 1
                for connection, timestamp, data in reader.messages(connections=connections, start=begin, stop=stop):
                    msgdef = connection.msgdef
                    yield RawMessage(
                        topic=connection.topic, msgtype=connection.msgtype, encoding='ros1',
                        schema_name=connection.msgtype.replace('/msg/', '/'), schema_encoding='ros1msg',
                        schema_data=getattr(msgdef, 'data', msgdef).encode(),
                        log_time=timestamp, publish_time=timestamp, data=data)
            return

        from mcap.reader import make_reader
        with open(self.filepath, 'rb') as file:
            reader = make_reader(file)
            stop = None if end is None else end + 1
            for schema, channel, message in reader.iter_messages(topics=topics, start_time=begin, end_time=stop,
                                                                 log_time_order=True):
                yield RawMessage(
                    topic=channel.topic, msgtype=_normalize_type(schema.name) if schema else '',
                    encoding=channel.message_encoding, schema_name=schema.name if schema else '',
                    schema_encoding=schema.encoding if schema else '', schema_data=schema.data if schema else b'',
                    log_time=message.log_time, publish_time=message.publish_time, data=message.data)

    def first_log_time(self, topic: str) -> Optional[int]:
        for message in self.messages([topic]):
            return message.log_time
        return None

    def _typestore(self, encoding: str):
        if encoding not in self._typestores:
            from rosbags.typesys import Stores, get_typestore
            self._typestores[encoding] = get_typestore(Stores.ROS1_NOETIC if encoding == 'ros1' else Stores.ROS2_FOXY)
        return self._typestores[encoding]

    def deserialize(self, message: RawMessage):
        if message.encoding == 'ros1':
            return self._typestore('ros1').deserialize_ros1(message.data, message.msgtype)
        if message.encoding == 'cdr':
            return self._typestore('cdr').deserialize_cdr(message.data, message.msgtype)
        raise ValueError(f'Topic "{message.topic}": unsupported message encoding "{message.encoding}"')

    def serialize(self, msg, like: RawMessage) -> bytes:
        """`msg` serialized the same way as the message `like` was."""
        if like.encoding == 'ros1':
            return bytes(self._typestore('ros1').serialize_ros1(msg, like.msgtype))
        return bytes(self._typestore('cdr').serialize_cdr(msg, like.msgtype))


class McapOutput:
    """MCAP file the messages are written into, with the schemas and channels of their source."""

    def __init__(self, filepath, profile: str = ''):
        from mcap.writer import Writer
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, 'wb')
        self._writer = Writer(self._file)
        self._writer.start(profile=profile, library='lidar-deskew')
        self._schemas: Dict[Tuple[str, str, bytes], int] = {}
        self._channels: Dict[Tuple[str, str, int], int] = {}
        self._sequence: Dict[int, int] = {}

    def write(self, like: RawMessage, data: bytes, topic: Optional[str] = None,
              log_time: Optional[int] = None) -> None:
        """Writes `data` on the topic (default: that of `like`) with the schema and encoding of `like`."""
        topic = topic or like.topic
        schema_key = (like.schema_name, like.schema_encoding, like.schema_data)
        if schema_key not in self._schemas:
            self._schemas[schema_key] = self._writer.register_schema(
                name=like.schema_name, encoding=like.schema_encoding, data=like.schema_data)
        schema_id = self._schemas[schema_key]
        channel_key = (topic, like.encoding, schema_id)
        if channel_key not in self._channels:
            self._channels[channel_key] = self._writer.register_channel(
                topic=topic, message_encoding=like.encoding, schema_id=schema_id)
        channel_id = self._channels[channel_key]
        sequence = self._sequence.get(channel_id, 0)
        self._sequence[channel_id] = sequence + 1
        self._writer.add_message(channel_id, log_time=like.log_time if log_time is None else log_time, data=data,
                                 publish_time=like.publish_time, sequence=sequence)

    def close(self) -> None:
        self._writer.finish()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
