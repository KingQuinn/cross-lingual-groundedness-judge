"""
Tables and hypothesis tests from results/preds/*.jsonl (and results/fertility.json).

    python scripts/analyze.py
Writes results/summary.json, results/RESULTS.md and the README results block.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge.construct import ITEM_TYPES  # noqa: E402
from judge.metrics import (coverage_at_precision, critical_metrics, explanation_metrics,  # noqa: E402
                           paired_bootstrap_diff, prf, slice_report)

# Primary condition first: English source -> Pidgin / Hausa answer
COND_ORDER = [("pcm", "en"), ("ha", "en"), ("pcm", "pcm"), ("ha", "ha"), ("en", "en")]
COND_NAME = {("pcm", "en"): "Pidgin ← English source", ("ha", "en"): "Hausa ← English source",
             ("pcm", "pcm"): "Pidgin ← Pidgin source", ("ha", "ha"): "Hausa ← Hausa source",
             ("en", "en"): "English ← English source"}
VARIANT_OF = {"no_hooks": "ha", "pcm_spelling": "pcm"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def orig(rows):
    return [r for r in rows if r.get("variant", "original") == "original"]


def cond(rows, lang, src):
    return [r for r in rows if r["lang"] == lang and r["source_lang"] == src]


def keyed(rows, with_src=False):
    return [{**r, "id": f"{r['doc_id']}|{r['item_type']}" + (f"|{r['source_lang']}" if with_src else "")} for r in rows]


def summarize(rows_all):
    rows = orig(rows_all)
    s = {"conditions": {}, "by_item_type": {}, "hypotheses": {}, "robustness": {}, "critical": {}}
    for c in COND_ORDER:
        rs = cond(rows, *c)
        if rs:
            s["conditions"][COND_NAME[c]] = slice_report(rs, keys=("lang",), n_boot=1000)[0]
            s["critical"][COND_NAME[c]] = critical_metrics(rs)
    for t in ITEM_TYPES:
        rs = [r for r in rows if r["item_type"] == t]
        if rs:
            s["by_item_type"][t] = prf(rs)["accuracy"]
    en = keyed(cond(rows, "en", "en"))
    for lang in ("pcm", "ha"):
        same, cross = keyed(cond(rows, lang, lang)), keyed(cond(rows, lang, "en"))
        if en and same:
            s["hypotheses"][f"H1 English vs {lang} (same-language source)"] = paired_bootstrap_diff(en, same)
        if en and cross:
            s["hypotheses"][f"H1 English vs {lang} (English source)"] = paired_bootstrap_diff(en, cross)
        if same and cross:
            s["hypotheses"][f"H4 same-language vs cross-lingual, {lang}"] = paired_bootstrap_diff(same, cross)
    for var, lang in VARIANT_OF.items():
        v = keyed([r for r in rows_all if r.get("variant") == var], with_src=True)
        o = keyed([r for r in rows if r["lang"] == lang], with_src=True)
        o = [r for r in o if r["id"] in {x["id"] for x in v}]
        if v and o:
            # accuracy, not F1: a variant set may contain only PASS items
            s["robustness"][var] = {"n": len(v), "acc_original": prf(o)["accuracy"], "acc_variant": prf(v)["accuracy"],
                                    "diff": paired_bootstrap_diff(o, v, stat="accuracy")}
    s["explanation"] = explanation_metrics(rows)
    s["overall"] = prf(rows)
    s["latency_s"] = float(np.mean([r["latency_s"] for r in rows]))
    s["coverage_at_95"] = coverage_at_precision(rows, 0.95)
    return s


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def render(summary, fertility):
    J = list(summary)
    L = ["### 1. F1 for catching unfaithful answers (FAIL class), test set, 95% bootstrap CI", "",
         "| Judge | " + " | ".join(COND_NAME[c] for c in COND_ORDER) + " |", "|---|" + "---:|" * len(COND_ORDER)]
    for j in J:
        cs = summary[j]["conditions"]
        cell = lambda m: f"{100*m['f1']:.1f} [{100*m['f1_lo']:.1f}, {100*m['f1_hi']:.1f}]"
        L.append(f"| {j} | " + " | ".join(cell(cs[COND_NAME[c]]) if COND_NAME[c] in cs else "–" for c in COND_ORDER) + " |")

    L += ["", "### 2. Safety: recall on CRITICAL errors (wrong dose / frequency / duration / threshold, swapped condition, unrelated answer)", "",
          "| Judge | " + " | ".join(COND_NAME[c] for c in COND_ORDER) + " | Severity agreement |", "|---|" + "---:|" * (len(COND_ORDER) + 1)]
    for j in J:
        cr = summary[j]["critical"]
        agree = [cr[k]["severity_agreement"] for k in cr if cr[k]["severity_agreement"] is not None]
        L.append(f"| {j} | " + " | ".join(pct(cr[COND_NAME[c]]["critical_recall"]) if COND_NAME[c] in cr else "–"
                                         for c in COND_ORDER) + f" | {pct(np.mean(agree)) if agree else '–'} |")

    L += ["", "### 3. Hypotheses (difference in F1 points, paired bootstrap 95% CI)", "",
          "| Judge | Comparison | Difference | CI | CI excludes 0? |", "|---|---|---:|---|---|"]
    for j in J:
        for name, (pt, lo, hi) in summary[j]["hypotheses"].items():
            L.append(f"| {j} | {name} | {100*pt:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")

    L += ["", "### 4. Accuracy by item type (original items, all conditions)", "",
          "| Judge | " + " | ".join(ITEM_TYPES) + " |", "|---|" + "---:|" * len(ITEM_TYPES)]
    for j in J:
        bt = summary[j]["by_item_type"]
        L.append(f"| {j} | " + " | ".join(pct(bt.get(t)) for t in ITEM_TYPES) + " |")

    ex = [j for j in J if summary[j]["explanation"]]
    if ex:
        L += ["", "### 5. Explanation quality (reasoning-mode judges)", "",
              "| Judge | Format complete | Evidence quoted from source | Evidence = gold sentence | Claim chrF vs gold English | Names both values (number errors) | Verdict ↔ error type consistent | Written = scored verdict |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for j in ex:
            e = summary[j]["explanation"]
            chrf_cell = "–" if e["claim_chrf"] is None else f"{e['claim_chrf']:.1f}"
            L.append(f"| {j} | {pct(e['format_complete'])} | {pct(e['evidence_valid'])} | {pct(e['evidence_correct'])} | "
                     f"{chrf_cell} | {pct(e['names_values'])} | {pct(e['consistent'])} | {pct(e['verdict_agrees_with_scoring'])} |")

    rb = [j for j in J if summary[j]["robustness"]]
    if rb:
        L += ["", "### 6. Robustness to how people actually type (accuracy, original → variant, paired)", "",
              "| Judge | Hausa without hooked letters | Pidgin spelling variants |", "|---|---|---|"]
        for j in rb:
            r = summary[j]["robustness"]
            cell = lambda v: (f"{100*r[v]['acc_original']:.1f} → {100*r[v]['acc_variant']:.1f} (n={r[v]['n']}) "
                              f"(Δ {-100*r[v]['diff'][0]:+.1f}, CI [{-100*r[v]['diff'][2]:+.1f}, {-100*r[v]['diff'][1]:+.1f}])") if v in r else "–"
            L.append(f"| {j} | {cell('no_hooks')} | {cell('pcm_spelling')} |")

    L += ["", "### 7. Cost and deployment view", "",
          "| Judge | Mean latency per judgement (s) | Auto-approved at 95% precision |", "|---|---:|---:|"]
    for j in J:
        cov = summary[j]["coverage_at_95"]
        L.append(f"| {j} | {summary[j]['latency_s']:.2f} | {pct(cov['coverage']) + '%' if cov else '–'} |")

    if fertility:
        langs = sorted({l for v in fertility.values() for l in v})
        L += ["", "### 8. Tokenizer fertility (tokens per word, lower is better)", "",
              "| Model | " + " | ".join(langs) + " |", "|---|" + "---:|" * len(langs)]
        for m, v in fertility.items():
            L.append(f"| {m} | " + " | ".join(f"{v.get(l, float('nan')):.2f}" for l in langs) + " |")
    return "\n".join(L) + "\n"


def main():
    files = sorted(glob.glob("results/preds/*.jsonl"))
    if not files:
        raise SystemExit("no predictions in results/preds/")
    summary = {}
    for f in files:
        name = os.path.basename(f)[:-6]
        summary[name] = summarize(load(f))
        print(f"{name}: n={summary[name]['overall']['n']} overall F1={summary[name]['overall']['f1']:.3f}")
    fert = json.load(open("results/fertility.json")) if os.path.exists("results/fertility.json") else None
    json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
    md = render(summary, fert)
    open("results/RESULTS.md", "w").write(md)
    if os.path.exists("README.md"):
        s = open("README.md").read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in s and b in s:
            open("README.md", "w").write(s[: s.index(a) + len(a)] + "\n" + md + s[s.index(b):])
            print("README.md results block updated")


if __name__ == "__main__":
    main()
