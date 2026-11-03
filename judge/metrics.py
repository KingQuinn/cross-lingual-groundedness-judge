"""
Metrics (PROTOCOL.md section 7).

A prediction row is a dict merged from the gold item and the judge output:
    label, item_type, lang, source_lang, source, answer, severity, gloss_en, meta,
    pred (verdict or None if unparseable), p_supported (or None),
    pred_generated, format_complete, pred_error_type, pred_severity,
    pred_claim_en, pred_evidence, pred_discrepancy, latency_s
Unparseable outputs are never dropped: they count as wrong.
"""

import random
import re

import numpy as np

POS = "unsupported"


def effective_pred(r):
    """Parse failure -> the opposite of gold (always wrong)."""
    if r["pred"] in ("supported", "unsupported"):
        return r["pred"]
    return "supported" if r["label"] == "unsupported" else "unsupported"


def prf(rows):
    tp = fp = fn = tn = 0
    for r in rows:
        p, g = effective_pred(r) == POS, r["label"] == POS
        tp += p and g
        fp += p and not g
        fn += (not p) and g
        tn += (not p) and (not g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"n": len(rows), "precision": prec, "recall": rec, "f1": f1,
            "balanced_acc": (tpr + tnr) / 2, "accuracy": (tp + tn) / max(1, len(rows)),
            "parse_fail": sum(r["pred"] not in ("supported", "unsupported") for r in rows) / max(1, len(rows))}


def bootstrap_ci(rows, stat="f1", n_boot=1000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        vals.append(prf(sample)[stat])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(rows_a, rows_b, stat="f1", n_boot=1000, seed=0):
    """Point estimate and 95% CI for stat(A) - stat(B).
    Rows are paired by 'id'; only pairs present in both sets are used (for the
    point estimate too). With fewer than 20 pairs, the sets are resampled
    independently instead."""
    rng = random.Random(seed)
    ia = {r["id"]: r for r in rows_a}
    ib = {r["id"]: r for r in rows_b}
    common = sorted(set(ia) & set(ib))
    diffs = []
    if len(common) >= 20:
        a0, b0 = [ia[i] for i in common], [ib[i] for i in common]
        point = prf(a0)[stat] - prf(b0)[stat]
        for _ in range(n_boot):
            ids = [common[rng.randrange(len(common))] for _ in common]
            diffs.append(prf([ia[i] for i in ids])[stat] - prf([ib[i] for i in ids])[stat])
    else:
        point = prf(rows_a)[stat] - prf(rows_b)[stat]
        for _ in range(n_boot):
            a = [rows_a[rng.randrange(len(rows_a))] for _ in rows_a]
            b = [rows_b[rng.randrange(len(rows_b))] for _ in rows_b]
            diffs.append(prf(a)[stat] - prf(b)[stat])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return point, float(lo), float(hi)


def ece(rows, n_bins=10):
    """Expected calibration error of confidence in the predicted verdict."""
    pairs = []
    for r in rows:
        if r.get("p_supported") is None or r["pred"] not in ("supported", "unsupported"):
            continue
        conf = r["p_supported"] if r["pred"] == "supported" else 1 - r["p_supported"]
        pairs.append((conf, r["pred"] == r["label"]))
    if not pairs:
        return None
    conf = np.array([c for c, _ in pairs])
    corr = np.array([k for _, k in pairs], dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(conf[bins == b].mean() - corr[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def coverage_at_precision(rows, target=0.95):
    """Product view: auto-approve answers with p_supported >= t. Returns the
    smallest t whose approved set has precision(supported) >= target, and the
    share of all answers auto-approved (coverage)."""
    scored = [r for r in rows if r.get("p_supported") is not None]
    if not scored:
        return None
    best = None
    for t in sorted({r["p_supported"] for r in scored}, reverse=True):
        approved = [r for r in scored if r["p_supported"] >= t]
        prec = sum(r["label"] == "supported" for r in approved) / len(approved)
        if prec >= target:
            best = {"threshold": t, "precision": prec, "coverage": len(approved) / len(rows)}
    return best or {"threshold": None, "precision": None, "coverage": 0.0}


def slice_report(rows, keys=("lang", "source_lang"), n_boot=1000):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        m = prf(rs)
        lo, hi = bootstrap_ci(rs, "f1", n_boot) if n_boot else (None, None)
        out.append({**dict(zip(keys, g)), **m, "f1_lo": lo, "f1_hi": hi, "ece": ece(rs)})
    return out


# --------------------------------------------------------------------------
# Explanation quality (PROTOCOL.md section 7.2): is the reasoning checkable and right?
# --------------------------------------------------------------------------
def _norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().strip('"\u201c\u201d').lower())


def explanation_metrics(rows):
    """Only rows with reasoning output (pred_claim_en key present).
    - format_complete: all five steps and the verdict present
    - evidence_valid: quoted evidence occurs word for word in the source
    - evidence_correct: quoted evidence equals the gold evidence sentence (supported/number/entity items)
    - claim_chrf: chrF of the judge's English claim vs the gold English gloss (cross-lingual understanding)
    - names_values: for number errors flagged FAIL, the discrepancy mentions both the wrong and the right value
    - consistent: the written verdict agrees with the written error type (FAIL <=> error type != none)
    - verdict_agrees_with_scoring: written verdict == scored verdict"""
    rs = [r for r in rows if "pred_claim_en" in r]
    if not rs:
        return None
    from .llm_stage import chrf
    out = {"n": len(rs), "format_complete": np.mean([bool(r.get("format_complete")) for r in rs])}
    ev = [r for r in rs if r.get("pred_evidence") and _norm(r["pred_evidence"]) not in ("none found", "none")]
    out["evidence_valid"] = float(np.mean([_norm(r["pred_evidence"]) in _norm(r["source"]) for r in ev])) if ev else None
    ec = [r for r in ev if r.get("evidence") and r["evidence"] != "none found"]
    out["evidence_correct"] = float(np.mean([_norm(r["pred_evidence"]) == _norm(r["evidence"]) for r in ec])) if ec else None
    cl = [r for r in rs if r.get("pred_claim_en") and r.get("gloss_en")]
    out["claim_chrf"] = float(np.mean([chrf(r["pred_claim_en"], r["gloss_en"]) for r in cl])) if cl else None
    nv = [r for r in rs if r["item_type"] in ("number", "number_swap") and r.get("pred") == "unsupported"
          and r.get("pred_discrepancy")]
    out["names_values"] = float(np.mean([
        r["meta"]["new"] in r["pred_discrepancy"] and r["meta"]["orig"] in r["pred_discrepancy"] for r in nv])) if nv else None
    co = [r for r in rs if r.get("pred_generated") and r.get("pred_error_type")]
    out["consistent"] = float(np.mean([(r["pred_generated"] == "unsupported") == (r["pred_error_type"] != "none")
                                       for r in co])) if co else None
    ag = [r for r in rs if r.get("pred_generated")]
    out["verdict_agrees_with_scoring"] = float(np.mean([r["pred_generated"] == r["pred"] for r in ag])) if ag else None
    return out


def critical_metrics(rows):
    """Safety view: recall on gold-CRITICAL failures (the number that matters
    most), and how often the judge's severity matches gold on caught failures."""
    crit = [r for r in rows if r.get("severity") == "CRITICAL"]
    caught = [r for r in rows if r.get("severity") in ("CRITICAL", "MINOR") and effective_pred(r) == "unsupported"
              and r.get("pred_severity")]
    return {
        "n_critical": len(crit),
        "critical_recall": float(np.mean([effective_pred(r) == "unsupported" for r in crit])) if crit else None,
        "severity_agreement": float(np.mean([r["pred_severity"] == r["severity"] for r in caught])) if caught else None,
    }
