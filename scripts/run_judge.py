"""
Run one judge over a split and save predictions (resumable).

    python scripts/run_judge.py --judge shortcut
    python scripts/run_judge.py --judge hf:CohereLabs/tiny-aya-earth --limit 500
    python scripts/run_judge.py --judge hf:Qwen/Qwen3-4B --mode verdict --name qwen4b-verdict
    python scripts/run_judge.py --judge gemini:gemini-3.8-flash
    python scripts/run_judge.py --judge hf:<base> --adapter checkpoints/<run>/final --name <run>

Writes <out_dir>/<name>.jsonl: one line per item (gold fields + prediction).
--limit samples whole documents (seed 0), identical for every judge, so
comparisons across languages and conditions stay paired.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge.construct import read_jsonl  # noqa: E402
from judge.prompt import build_prompt_text, parse_output  # noqa: E402

GOLD_KEYS = ("id", "doc_id", "split", "lang", "source_lang", "item_type", "label", "question", "source", "answer",
             "error_type", "severity", "gloss_en", "evidence", "meta", "hard", "variant")


def pick_items(items, split, limit):
    """Sample whole DOCUMENTS (seed 0) until about `limit` original items are
    covered, keeping every condition and robustness copy of those documents.
    Every judge gets the same documents, and comparisons across languages and
    conditions stay paired (same document, same item type)."""
    rows = [i for i in items if i["split"] == split]
    if not limit:
        return rows
    per_doc = {}
    for i in rows:
        if i.get("variant", "original") == "original":
            per_doc[i["doc_id"]] = per_doc.get(i["doc_id"], 0) + 1
    docs = sorted(per_doc)
    random.Random(0).shuffle(docs)
    keep, n = set(), 0
    for d in docs:
        if n >= limit:
            break
        keep.add(d)
        n += per_doc[d]
    return [i for i in rows if i["doc_id"] in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="shortcut | hf:<model_id> | gemini:<model>")
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="reasoning", choices=["reasoning", "verdict"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_variants", action="store_true", help="skip robustness copies")
    ap.add_argument("--out_dir", default="results/preds")
    a = ap.parse_args()

    items = pick_items(read_jsonl(a.items), a.split, a.limit)
    if a.no_variants:
        items = [i for i in items if i.get("variant", "original") == "original"]
    name = a.name or a.judge.replace("/", "_").replace(":", "-") + ("" if a.mode == "reasoning" else "-verdict")
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{name}.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{name}: {len(items)} items, {len(done)} already done, {len(todo)} to run")

    kind, _, model_id = a.judge.partition(":")
    if kind == "shortcut":
        from judge.baselines import shortcut_predict

        def run(it):
            t0 = time.time()
            v = shortcut_predict(it)["verdict"]
            return {"pred": v, "p_supported": None, "pred_generated": v, "latency_s": time.time() - t0,
                    "mode": "rule"}
    elif kind == "hf":
        from judge.hf_judge import judge_item, load_model
        model, tok = load_model(model_id, load_4bit=not a.no_4bit, adapter=a.adapter)
        run = lambda it: judge_item(model, tok, it, mode=a.mode)
    elif kind == "gemini":
        from judge.llm_stage import GeminiClient
        client = GeminiClient(model=model_id, min_interval=4.0)

        def run(it):
            t0 = time.time()
            text = client.generate(build_prompt_text(it, a.mode), json_mode=False)
            p = parse_output(text)
            row = {"pred": p["verdict"], "p_supported": None, "pred_generated": p["verdict"], "raw": text,
                   "latency_s": time.time() - t0, "mode": a.mode}
            if a.mode == "reasoning":
                row.update(format_complete=p["complete"], pred_error_type=p["error_type"],
                           pred_severity=p["severity"], pred_claim_en=p["claim_en"],
                           pred_evidence=p["evidence"], pred_discrepancy=p["discrepancy"])
            return row
    else:
        raise SystemExit(f"unknown judge {a.judge}")

    with open(path, "a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            row = {**{key: it.get(key) for key in GOLD_KEYS}, **run(it), "judge": name}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
