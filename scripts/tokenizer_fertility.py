"""
Tokenizer fertility: how many tokens each model needs per word, per language.
Higher fertility = longer inputs, more memory and cost, and often weaker
modelling of that language. Measured on the test split's supported answers
and same-language source passages.

    python scripts/tokenizer_fertility.py --models CohereLabs/tiny-aya-earth Qwen/Qwen3-4B meta-llama/Llama-3.2-3B-Instruct
Writes results/fertility.json (analyze.py renders it into the README).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge.construct import read_jsonl  # noqa: E402
from judge.text import normalize_text  # noqa: E402


def texts_by_lang(items):
    out = {}
    for it in items:
        if it["split"] == "test" and it["item_type"] == "supported" and it.get("variant", "original") == "original" \
                and it["lang"] == it["source_lang"]:
            out.setdefault(it["lang"], []).extend([normalize_text(it["answer"]), normalize_text(it["source"])])
    return out


def fertility(count_tokens, texts):
    """Total tokens / total whitespace-separated words."""
    words = sum(len(t.split()) for t in texts)
    return sum(count_tokens(t) for t in texts) / max(1, words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out", default="results/fertility.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    texts = texts_by_lang(read_jsonl(a.items))
    res = {}
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        count = lambda t: len(tok(t, add_special_tokens=False).input_ids)
        res[m] = {lang: round(fertility(count, ts), 3) for lang, ts in sorted(texts.items())}
        print(m, res[m])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
