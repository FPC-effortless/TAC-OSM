"""Label-separated, document-owned raw-byte observations; no model inference."""
from dataclasses import dataclass
from pathlib import PurePosixPath


def encode_bytes(raw: bytes) -> tuple[int, ...]:
    if not isinstance(raw,bytes):raise ValueError('original immutable bytes required')
    return tuple(raw)


def decode_bytes(tokens) -> bytes:
    values=tuple(tokens)
    if any(type(x) is not int or not 0<=x<256 for x in values):
        raise ValueError('integer byte IDs0..255 required')
    return bytes(values)


@dataclass(frozen=True)
class ByteDocument:
    owner: str
    raw: bytes

    def __post_init__(self):
        path=PurePosixPath(self.owner)
        if (not self.owner or path.is_absolute() or '..' in path.parts
            or self.owner!=path.as_posix() or '\\' in self.owner):
            raise ValueError('canonical source owner required')
        if not isinstance(self.raw,bytes) or len(self.raw)<2:
            raise ValueError('at least two immutable original bytes required')


@dataclass(frozen=True)
class ByteObservation:
    owner: str
    first_position: int
    last_position: int
    tokens: tuple[int,...]


def _position(document,position):
    if type(position) is not int or not 0<=position<len(document.raw)-1:
        raise ValueError('observed source position with genuine next byte required')


def observe_at(document: ByteDocument, position: int, *, history: int=128) -> ByteObservation:
    _position(document,position)
    if type(history) is not int or history<1:raise ValueError('positive history length required')
    first=max(0,position-history+1)
    return ByteObservation(document.owner,first,position,encode_bytes(document.raw[first:position+1]))


def next_target(document: ByteDocument, position: int) -> int:
    _position(document,position)
    return document.raw[position+1]
