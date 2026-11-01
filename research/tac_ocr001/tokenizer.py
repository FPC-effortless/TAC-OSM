"""Lossless UTF-8 bytes; no learned/adaptive tokenizer or text normalization."""

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259


def encode(text: str, *, boundaries: bool = True) -> list[int]:
    ids = list(text.encode("utf-8"))
    return [BOS, *ids, EOS] if boundaries else ids


def decode(ids, *, errors="strict") -> str:
    values = list(ids)
    if any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in values):
        raise ValueError("invalid UTF-8 byte token")
    return bytes(x for x in values if x < 256).decode("utf-8", errors=errors)
