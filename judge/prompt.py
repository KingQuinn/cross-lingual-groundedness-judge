"""
The single judge prompt used for every judge (PROTOCOL.md section 2), the
fine-tuning target, and the parser.

Two output modes:
  reasoning  five plain-language steps, then the verdict (the main judge)
  verdict    the verdict line only (ablation: what does the reasoning buy?)
"""

import re

from .text import normalize_text

RUBRIC = """You check whether a health assistant's ANSWER is faithful to a trusted SOURCE text. The source may be in English while the answer is in Nigerian Pidgin or Hausa; compare the meaning across languages.

Rubric:
1. PASS only if every factual claim in the answer can be found in, or directly follows from, the source. Paraphrase, translation, and numbers written as words (e.g. 'two' for '2') are fine.
2. Judge faithfulness, not completeness: leaving out details from the source is fine.
3. Do not use outside knowledge: a claim that is true in the world but not in the source FAILS.
4. General advice to see a doctor or health worker is not a factual claim and does not cause a FAIL.
5. One unsupported or changed claim makes the whole answer FAIL."""

FORMAT_REASONING = """Write your evaluation in exactly this format:
1. Claim (in English): <the answer's factual claims, translated into English>
2. Source evidence: "<the source sentence the claim depends on, copied word for word>" or none found
3. Discrepancy: <exactly what differs, e.g. 'The answer says 6.4 percent, but the source says 5.7.' If nothing differs, write 'None.' and say why the answer is still faithful>
4. Error type: <none | number | entity | wrong_source | other>
5. Severity: <CRITICAL if a dose, frequency, duration, age or clinical threshold is wrong, a medicine or condition is swapped, or the answer is unrelated to the source; MINOR for other unsupported content; NONE if PASS>
Verdict: <PASS or FAIL>"""

FORMAT_VERDICT = """Reply with exactly one line:
Verdict: <PASS or FAIL>"""

TEMPLATE = """SOURCE:
{source}

QUESTION:
{question}

ANSWER:
{answer}"""

ERROR_TYPES = ("none", "number", "entity", "wrong_source", "other")
SEVERITIES = ("NONE", "MINOR", "CRITICAL")


def build_messages(item, mode="reasoning"):
    system = RUBRIC + "\n\n" + (FORMAT_REASONING if mode == "reasoning" else FORMAT_VERDICT)
    user = TEMPLATE.format(source=normalize_text(item["source"]), question=normalize_text(item["question"]),
                           answer=normalize_text(item["answer"]))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt_text(item, mode="reasoning"):
    """Single-string version for APIs."""
    m = build_messages(item, mode)
    return m[0]["content"] + "\n\n" + m[1]["content"]


def target_text(item, mode="reasoning"):
    """Gold output for fine-tuning, built from the item's construction record."""
    verdict = "PASS" if item["label"] == "supported" else "FAIL"
    if mode == "verdict":
        return f"Verdict: {verdict}"
    ev = item["evidence"]
    ev = ev if ev == "none found" else f'"{ev}"'
    return (f"1. Claim (in English): {item['gloss_en']}\n"
            f"2. Source evidence: {ev}\n"
            f"3. Discrepancy: {item['discrepancy']}\n"
            f"4. Error type: {item['error_type']}\n"
            f"5. Severity: {item['severity']}\n"
            f"Verdict: {verdict}")


_FIELDS = {
    "claim_en": r"1\.\s*Claim[^:]*:\s*(.+)",
    "evidence": r"2\.\s*Source evidence\s*:\s*(.+)",
    "discrepancy": r"3\.\s*Discrepancy\s*:\s*(.+)",
    "error_type": r"4\.\s*Error type\s*:\s*([A-Za-z_]+)",
    "severity": r"5\.\s*Severity\s*:\s*([A-Za-z]+)",
}


def parse_output(text):
    """Returns dict with verdict ('supported'/'unsupported' or None) and the
    reasoning fields (None when missing). `complete` is True when every step
    and the verdict were found."""
    if not text:
        return {"verdict": None, "complete": False, **{k: None for k in _FIELDS}}
    out = {}
    for k, pat in _FIELDS.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        out[k] = m.group(1).strip() if m else None
    if out["evidence"]:
        out["evidence"] = out["evidence"].strip().strip('"“”').strip()
    if out["error_type"]:
        out["error_type"] = out["error_type"].lower() if out["error_type"].lower() in ERROR_TYPES else "other"
    if out["severity"]:
        out["severity"] = out["severity"].upper() if out["severity"].upper() in SEVERITIES else None
    m = re.search(r"Verdict\s*:\s*\**\s*(PASS|FAIL)", text, flags=re.IGNORECASE)
    out["verdict"] = None if not m else ("supported" if m.group(1).upper() == "PASS" else "unsupported")
    out["complete"] = out["verdict"] is not None and all(out[k] for k in _FIELDS)
    return out
