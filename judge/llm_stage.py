"""
LLM stage of data construction (PROTOCOL.md 5.2): rewrite English answers,
translate into Pidgin / Hausa, back-translate, filter.

All calls are batched (several items per request, JSON in / JSON out) and
cached on disk, so a run interrupted by rate limits resumes where it stopped.
"""

import hashlib
import json
import os
import re
import time

from .text import normalize_text, same_numbers

LANG_NAMES = {"pcm": "Nigerian Pidgin (Naija)", "ha": "Hausa", "en": "English"}


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------
class GeminiClient:
    """Thin wrapper around the google-genai SDK with retry and pacing."""

    def __init__(self, model="gemini-2.5-flash", api_key=None, min_interval=4.0, max_retries=6):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model, self.min_interval, self.max_retries = model, min_interval, max_retries
        self._last = 0.0

    def generate(self, prompt, json_mode=True):
        cfg = self._types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json" if json_mode else "text/plain")
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.time()
                return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text
            except Exception as e:  # rate limits, transient server errors
                delay = min(60, 5 * 2**attempt)
                print(f"  LLM error ({type(e).__name__}: {str(e)[:120]}), retrying in {delay}s")
                time.sleep(delay)
        raise RuntimeError("LLM call failed after retries")


class CachedClient:
    """Disk cache keyed by the prompt hash."""

    def __init__(self, client, path):
        self.client, self.path = client, path
        self.cache = json.load(open(path)) if os.path.exists(path) else {}

    def generate(self, prompt, json_mode=True):
        k = hashlib.sha256(prompt.encode()).hexdigest()
        if k not in self.cache:
            self.cache[k] = self.client.generate(prompt, json_mode=json_mode)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        return self.cache[k]


# --------------------------------------------------------------------------
# Batched JSON calls
# --------------------------------------------------------------------------
def parse_json_list(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return data


def batched_call(client, instructions, payload, out_key, batch_size=8):
    """payload: list of dicts with an 'id'. Returns {id: output string}."""
    out = {}
    for i in range(0, len(payload), batch_size):
        batch = payload[i:i + batch_size]
        prompt = instructions + "\n\nINPUT:\n" + json.dumps(batch, ensure_ascii=False, indent=1)
        try:
            for row in parse_json_list(client.generate(prompt)):
                if isinstance(row, dict) and "id" in row and isinstance(row.get(out_key), str):
                    out[str(row["id"])] = row[out_key].strip()
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  batch {i // batch_size}: unparseable response ({e}); items skipped")
    return out


REWRITE_INSTR = """You are helping build a test set for checking health answers.
For each item, rewrite SENTENCE as a short, natural answer to QUESTION, as a health assistant would say it.
Rules:
- Keep every fact. Keep every number exactly as written, in digits.
- Do not add any information that is not in SENTENCE.
- One or two sentences.
Return a JSON list of objects: {"id": <id>, "answer": <answer>}."""

TRANSLATE_INSTR = """Translate each text into {lang}.
Rules:
- Keep every number in digits exactly as in the original.
- Keep disease and medicine names in English when there is no common {lang} word for them.
- Translate meaning faithfully; do not add or drop information.
Return a JSON list of objects: {{"id": <id>, "text": <translation>}}."""

BACK_INSTR = """Translate each text into English, as literally as possible while staying fluent.
Return a JSON list of objects: {"id": <id>, "text": <translation>}."""


def rewrite_answers(bases, client, batch_size=10):
    payload = [{"id": b.doc_id, "question": b.question, "sentence": b.sentence} for b in bases]
    out = batched_call(client, REWRITE_INSTR, payload, "answer", batch_size)
    kept = 0
    for b in bases:
        a = out.get(b.doc_id)
        if a and same_numbers(a, b.sentence):  # protocol: every number preserved
            b.answer["en"] = normalize_text(a)
            kept += 1
    return {"rewritten": len(out), "kept_numbers_preserved": kept}


def chrf(hyp, ref):
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def translate_bases(bases, lang, client, batch_size=10, chrf_min=50.0):
    """Translate every passage SENTENCE (kept aligned, so the gold evidence is
    an exact sentence of the translated source) and the English answer;
    back-translate; keep a base only if every number survives in every
    sentence and chrF(back-translation, original) >= chrf_min for both the
    whole passage and the answer."""
    todo = [b for b in bases if "en" in b.answer]
    payload = []
    for b in todo:
        payload += [{"id": f"{b.doc_id}|s{k}", "text": s} for k, s in enumerate(b.passage_sents)]
        payload.append({"id": f"{b.doc_id}|a", "text": b.answer["en"]})
    fwd = batched_call(client, TRANSLATE_INSTR.format(lang=LANG_NAMES[lang]), payload, "text", batch_size)
    back = batched_call(client, BACK_INSTR, [{"id": k, "text": v} for k, v in fwd.items()], "text", batch_size)

    stats = {"candidates": len(todo), "translated": 0, "fail_numbers": 0, "fail_chrf": 0, "kept": 0}
    for b in todo:
        keys = [f"{b.doc_id}|s{k}" for k in range(len(b.passage_sents))] + [f"{b.doc_id}|a"]
        if not all(k in fwd and k in back for k in keys):
            continue
        stats["translated"] += 1
        sents = [normalize_text(fwd[k]) for k in keys[:-1]]
        ans = normalize_text(fwd[keys[-1]])
        if not all(same_numbers(t, o) for t, o in zip(sents, b.passage_sents)) or not same_numbers(ans, b.answer["en"]):
            stats["fail_numbers"] += 1
            continue
        back_passage = " ".join(back[k] for k in keys[:-1])
        if min(chrf(back_passage, b.passage), chrf(back[keys[-1]], b.answer["en"])) < chrf_min:
            stats["fail_chrf"] += 1
            continue
        b.passage_tr[lang], b.answer[lang] = sents, ans
        stats["kept"] += 1
    return stats


class EchoClient:
    """Offline stand-in for tests and dry runs: 'rewrites' and 'translates' by
    returning the input text unchanged. Exercises the full pipeline without an API."""

    def generate(self, prompt, json_mode=True):
        batch = json.loads(prompt.split("INPUT:\n", 1)[1])
        key = "answer" if "rewrite SENTENCE" in prompt else "text"
        src = "sentence" if key == "answer" else "text"
        return json.dumps([{"id": r["id"], key: r[src]} for r in batch])
