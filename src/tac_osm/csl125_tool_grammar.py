"""Bounded ASCII/byte JSON-prefix constraint for CSL tool-call experiments.

This grammar proves *JSON syntax* and forces the function_calls envelope prefix.
It is NOT a full JSON-Schema-to-token grammar. The final strict tool-schema
validator must reject unknown tools, wrong fields and wrong argument types.
No performance or Needle-equivalent schema guarantee is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass

_HEADER = '{"function_calls":['
_CANDIDATES = tuple(range(32, 127))
_HEX = "0123456789abcdefABCDEF"


class _Incomplete(Exception):
    pass


class _Invalid(Exception):
    pass


@dataclass(frozen=True)
class JSONPrefixState:
    valid: bool
    complete: bool


def _scan_json(text: str) -> JSONPrefixState:
    n = len(text)

    def more(i: int) -> None:
        if i >= n:
            raise _Incomplete()

    def parse_string(i: int) -> int:
        more(i)
        if text[i] != '"':
            raise _Invalid()
        i += 1
        while True:
            more(i)
            c = text[i]
            if c == '"':
                return i + 1
            if ord(c) < 32 or ord(c) > 126:
                raise _Invalid()
            if c == '\\':
                i += 1
                more(i)
                if text[i] in '"/\\bfnrt':
                    i += 1
                    continue
                if text[i] != 'u':
                    raise _Invalid()
                for _ in range(4):
                    i += 1
                    more(i)
                    if text[i] not in _HEX:
                        raise _Invalid()
                i += 1
            else:
                i += 1

    def number(i: int) -> int:
        if text[i] == '-':
            i += 1
            more(i)
        if text[i] == '0':
            i += 1
            if i < n and text[i].isdigit():
                raise _Invalid()
        elif '1' <= text[i] <= '9':
            while i < n and '0' <= text[i] <= '9':
                i += 1
        else:
            raise _Invalid()
        if i < n and text[i] == '.':
            i += 1
            more(i)
            if not ('0' <= text[i] <= '9'):
                raise _Invalid()
            while i < n and '0' <= text[i] <= '9':
                i += 1
        if i < n and text[i] in 'eE':
            i += 1
            more(i)
            if text[i] in '+-':
                i += 1
                more(i)
            if not ('0' <= text[i] <= '9'):
                raise _Invalid()
            while i < n and '0' <= text[i] <= '9':
                i += 1
        return i

    def literal(i: int, expected: str) -> int:
        remaining = text[i:i+len(expected)]
        if expected.startswith(remaining) and len(remaining) < len(expected):
            raise _Incomplete()
        if remaining != expected:
            raise _Invalid()
        return i + len(expected)

    def value(i: int, depth: int) -> int:
        if depth > 12:
            raise _Invalid()
        more(i)
        c = text[i]
        if c == '"':
            return parse_string(i)
        if c == '{':
            i += 1
            more(i)
            if text[i] == '}':
                return i + 1
            while True:
                i = parse_string(i)
                more(i)
                if text[i] != ':':
                    raise _Invalid()
                i = value(i + 1, depth + 1)
                more(i)
                if text[i] == '}':
                    return i + 1
                if text[i] != ',':
                    raise _Invalid()
                i += 1
        if c == '[':
            i += 1
            more(i)
            if text[i] == ']':
                return i + 1
            while True:
                i = value(i, depth + 1)
                more(i)
                if text[i] == ']':
                    return i + 1
                if text[i] != ',':
                    raise _Invalid()
                i += 1
        if c in '-0123456789':
            return number(i)
        if c == 't':
            return literal(i, 'true')
        if c == 'f':
            return literal(i, 'false')
        if c == 'n':
            return literal(i, 'null')
        raise _Invalid()

    try:
        if not text:
            return JSONPrefixState(True, False)
        pos = value(0, 0)
        if pos < n:
            return JSONPrefixState(False, False)
        return JSONPrefixState(True, True)
    except _Incomplete:
        return JSONPrefixState(True, False)
    except _Invalid:
        return JSONPrefixState(False, False)


def prefix_state(raw: str) -> JSONPrefixState:
    if type(raw) is not str or len(raw) > 8192:
        return JSONPrefixState(False, False)
    if len(raw) < len(_HEADER):
        return JSONPrefixState(_HEADER.startswith(raw), False)
    if not raw.startswith(_HEADER):
        return JSONPrefixState(False, False)
    # Before schema validation, only JSON syntax and fixed envelope beginning
    # are guaranteed. Additional top-level keys are later rejected.
    return _scan_json(raw)


def allowed_byte_ids(prefix: str) -> tuple[int, ...]:
    state = prefix_state(prefix)
    if not state.valid or state.complete:
        return ()
    if len(prefix) < len(_HEADER):
        return (ord(_HEADER[len(prefix)]),)
    return tuple(c for c in _CANDIDATES if prefix_state(prefix + chr(c)).valid)
