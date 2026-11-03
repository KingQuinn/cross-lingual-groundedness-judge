"""Shortcut baseline (PROTOCOL.md section 6): flags an answer as unsupported if
it contains any number (digits or number words) that does not appear
literally in the source. No learning, no language knowledge beyond number words.

If a learned judge scores close to this, it may be doing the same thing.
hard_supported items are designed so this baseline gets them wrong."""

import re

from .text import NUMBER_WORDS, number_strings

_ALL_WORDS = {w for ws in NUMBER_WORDS.values() for w in ws}


def _tokens(text):
    return set(re.findall(r"[\wɗƙɓƴ']+", text.lower()))


def shortcut_predict(item):
    src_nums = set(number_strings(item["source"]))
    src_toks = _tokens(item["source"])
    novel_digits = [n for n in number_strings(item["answer"]) if n not in src_nums]
    novel_words = [w for w in _tokens(item["answer"]) & _ALL_WORDS if w not in src_toks]
    unsupported = bool(novel_digits or novel_words)
    return {"verdict": "unsupported" if unsupported else "supported",
            "error_type": "number" if unsupported else "none",
            "unsupported_span": (novel_digits or novel_words or [""])[0],
            "confidence": 1.0}
