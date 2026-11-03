"""
Convert RAGTruth (Niu et al., 2024; wandb/RAGTruth-processed, MIT) QA test
items into our item format: a natural-hallucination test set in English.

    python scripts/ragtruth_items.py --out data/ragtruth_qa_test.jsonl
    python scripts/run_judge.py --judge hf:<model> --items data/ragtruth_qa_test.jsonl --name <model>-ragtruth

An answer is `unsupported` (FAIL) if annotators marked at least one
hallucination span in it. Severity is not annotated in RAGTruth (UNKNOWN).
"""

import argparse
import ast
import json
import os


def parse_labels(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(x)


def convert(row):
    labels = parse_labels(row.get("hallucination_labels"))
    return {"id": f"ragtruth-{row['id']}", "doc_id": f"ragtruth-{row['id']}", "split": "test",
            "lang": "en", "source_lang": "en", "item_type": "natural_ragtruth",
            "label": "unsupported" if labels else "supported",
            "question": row["query"], "source": row["context"], "answer": row["output"],
            "error_type": "none" if not labels else "other", "severity": "UNKNOWN",
            "gloss_en": row["output"], "evidence": "", "meta": {"spans": labels}, "hard": False,
            "variant": "original"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ragtruth_qa_test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    rows = [convert(r) for r in ds if str(r.get("task_type", "")).lower() == "qa"]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_un = sum(r["label"] == "unsupported" for r in rows)
    print(f"{len(rows)} QA items ({n_un} with hallucinations) -> {a.out}")


if __name__ == "__main__":
    main()
