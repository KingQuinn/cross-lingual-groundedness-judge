"""
Build the label-by-construction dataset (PROTOCOL.md section 5).

    # real run (Colab): needs GEMINI_API_KEY
    python scripts/build_dataset.py --n_docs 300
    # offline dry run with a fake LLM (no translation; for testing the pipeline)
    python scripts/build_dataset.py --dry_run --medquad_jsonl tests/fixtures/medquad_sample.jsonl

Outputs in --out (default data/):
    bases.json          passages, answers, translations (reused by later steps)
    items.jsonl         every test item, all languages and conditions
    stats.json          counts, filter and skip rates (reported in the README)
    spotcheck_<lang>.csv  random sample for the fluent-reader translation check
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge.construct import make_items, robustness_variants, select_passages, split_by_document, write_jsonl  # noqa: E402
from judge.llm_stage import CachedClient, EchoClient, GeminiClient, rewrite_answers, translate_bases  # noqa: E402


def load_medquad(path=None):
    if path:
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    from datasets import load_dataset
    return [dict(r) for r in load_dataset("lavita/MedQuAD", split="train")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=300)
    ap.add_argument("--langs", nargs="+", default=["pcm", "ha"])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--chrf_min", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--medquad_jsonl", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_spotcheck", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    records = load_medquad(a.medquad_jsonl)
    bases = split_by_document(select_passages(records, a.n_docs, seed=a.seed), seed=a.seed)
    print(f"passages selected: {len(bases)}  splits: {Counter(b.split for b in bases)}")

    client = EchoClient() if a.dry_run else CachedClient(
        GeminiClient(model=a.model), os.path.join(a.out, "cache", "llm_cache.json"))
    stats = {"n_records": len(records), "n_passages": len(bases), "model": "echo" if a.dry_run else a.model,
             "rewrite": rewrite_answers(bases, client, a.batch_size)}
    print("rewrite:", stats["rewrite"])
    for lang in a.langs:
        stats[f"translate_{lang}"] = translate_bases(bases, lang, client, a.batch_size, a.chrf_min)
        print(f"translate {lang}:", stats[f"translate_{lang}"])

    # Keep only passages that survived every stage in every language, so all
    # conditions are compared on the SAME passages (no composition effects).
    common = [b for b in bases if "en" in b.answer and all(l in b.answer for l in a.langs)]
    stats["n_common_passages"] = len(common)
    with open(os.path.join(a.out, "bases.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in common], f, ensure_ascii=False, indent=1)

    # Primary condition: English source -> Pidgin/Hausa answer. Same-language
    # and English-English conditions are the comparisons (H1, H4).
    conditions = [(l, "en") for l in a.langs] + [(l, l) for l in a.langs] + [("en", "en")]
    items, stats["counts"] = [], {}
    for lang, src in conditions:
        its, counts = make_items(common, lang, src, seed=a.seed)
        items += its
        stats["counts"][f"{lang}<-{src}"] = counts
    variants = robustness_variants(items)
    items += variants
    stats["n_robustness_variants"] = dict(Counter(v.variant for v in variants))
    write_jsonl(items, os.path.join(a.out, "items.jsonl"))
    stats["n_items"] = len(items)
    stats["items_by"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(
        Counter((i.split, f"{i.lang}<-{i.source_lang}", i.item_type) for i in items if i.variant == "original").items())}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    rng = random.Random(a.seed)
    for lang in a.langs:
        sample = rng.sample(common, min(a.n_spotcheck, len(common)))
        with open(os.path.join(a.out, f"spotcheck_{lang}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "english_passage", f"{lang}_passage", "english_answer", f"{lang}_answer",
                        "meaning_preserved (y/n)", "notes"])
            for b in sample:
                w.writerow([b.doc_id, b.passage, " ".join(b.passage_tr[lang]), b.answer["en"], b.answer[lang], "", ""])
    print(f"items: {len(items)}  common passages: {len(common)}  -> {a.out}/")


if __name__ == "__main__":
    main()
