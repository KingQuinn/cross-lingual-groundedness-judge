"""
Label-by-construction test items with gold explanations (PROTOCOL.md section 5).

Every item carries, besides its PASS/FAIL label, the full gold reasoning trace:
  gloss_en     the answer's claims in English
  evidence     the source sentence the claim depends on, quoted from the source
  discrepancy  exactly what differs (or why it is still supported)
  error_type, severity
These are built from what we know about each construction, so they are correct
by construction and are used both as fine-tuning targets and to score the
judge's own explanations.

No LLM calls in this file; it is fully unit-tested.
"""

import json
import random
import re
from dataclasses import asdict, dataclass, field

from .text import (CONDITION_WORDS, NUMBER_WORDS, OPENER_WORDS, REFERRAL, clean_medquad, content_overlap, find_numbers,
                   context_window, is_critical_number, normalize_text, number_phrase, number_strings, pidgin_respell,
                   split_sentences, strip_hooks, word_count)

PASS_TYPES = ("supported", "hard_supported", "omission", "referral")
FAIL_TYPES = ("number", "number_swap", "entity", "wrong_source", "added_fact")
ITEM_TYPES = PASS_TYPES + FAIL_TYPES
MAX_OVERLAP = 0.5  # label-validity guard for added_fact and wrong_source
NO_EVIDENCE = "none found"


@dataclass
class Base:
    """One source passage and the sentence an assistant answer is built from."""
    doc_id: str
    focus: str
    question: str
    passage_sents: list           # English passage, as sentences
    sent_idx: int                 # index of the answer's source sentence
    split: str = ""
    answer: dict = field(default_factory=dict)       # lang -> answer text ("en" = rewrite)
    passage_tr: dict = field(default_factory=dict)   # lang -> list of translated sentences (aligned)

    @property
    def passage(self):
        return " ".join(self.passage_sents)

    @property
    def sentence(self):
        return self.passage_sents[self.sent_idx]

    def source(self, lang):
        return self.passage if lang == "en" else " ".join(self.passage_tr[lang])

    def evidence(self, lang):
        return self.sentence if lang == "en" else self.passage_tr[lang][self.sent_idx]


@dataclass
class Item:
    id: str
    doc_id: str
    split: str
    lang: str             # language of the answer
    source_lang: str      # language of the source passage shown to the judge
    item_type: str
    label: str            # "supported" (PASS) | "unsupported" (FAIL)
    question: str
    source: str
    answer: str
    error_type: str       # none | number | entity | wrong_source | other
    severity: str         # NONE | MINOR | CRITICAL
    gloss_en: str         # gold step 1
    evidence: str         # gold step 2 (a sentence of `source`, or "none found")
    discrepancy: str      # gold step 3
    meta: dict = field(default_factory=dict)   # e.g. {"orig": "5.7", "new": "6.4"}
    hard: bool = False    # wrong_source / added_fact: foreign text shares all its numbers with the source
    variant: str = "original"   # original | no_hooks | pcm_spelling (robustness copies, test only)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------
# 1. Passage selection and splits
# --------------------------------------------------------------------------
def select_passages(records, n_docs, seed=0, min_sents=2, max_sents=5, max_words=120):
    """At most one passage per document: the first window of 2-5 sentences
    (<= max_words) containing a numeric sentence of 6-40 words. The answer
    sentence prefers one that mentions the focus term (so entity swaps work)."""
    rng = random.Random(seed)
    by_doc = {}
    for r in records:
        if r.get("answer") and r.get("question_focus"):
            by_doc.setdefault(str(r["document_id"]), []).append(r)
    doc_ids = sorted(by_doc)
    rng.shuffle(doc_ids)
    bases = []
    for d in doc_ids:
        for r in by_doc[d]:
            b = _passage_from_record(d, r, min_sents, max_sents, max_words)
            if b:
                bases.append(b)
                break
        if len(bases) >= n_docs:
            break
    return bases


def _passage_from_record(doc_id, r, min_sents, max_sents, max_words):
    sents = [normalize_text(s) for s in split_sentences(clean_medquad(r["answer"]))]
    focus = r["question_focus"].strip()
    for i in range(len(sents)):
        for j in range(i + min_sents, min(i + max_sents, len(sents)) + 1):
            window = sents[i:j]
            if word_count(" ".join(window)) > max_words:
                break
            numeric = [k for k, s in enumerate(window) if number_strings(s) and 6 <= word_count(s) <= 40]
            if not numeric:
                continue
            with_focus = [k for k in numeric if focus.lower() in window[k].lower()]
            return Base(doc_id=doc_id, focus=focus, question=normalize_text(r["question"]),
                        passage_sents=window, sent_idx=(with_focus or numeric)[0])
    return None


def split_by_document(bases, seed=0, frac=(0.6, 0.2, 0.2)):
    rng = random.Random(seed)
    idx = list(range(len(bases)))
    rng.shuffle(idx)
    n_tr, n_va = int(frac[0] * len(idx)), int(frac[1] * len(idx))
    for k, i in enumerate(idx):
        bases[i].split = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
    return bases


# --------------------------------------------------------------------------
# 2. Constructions. Each returns enough detail to write the gold explanation.
# --------------------------------------------------------------------------
def _replace_number(text, value, new):
    """Replace the first occurrence of `value`; also return the quoted context
    before and after the change, for the explanation."""
    for s, e, t in find_numbers(text):
        if t == value:
            changed = text[:s] + new + text[e:]
            return changed, context_window(text, s, e), context_window(changed, s, s + len(new))
    return None


def perturb_number(answer, source, rng):
    """One integer changed to a value absent from source and answer -> (new_answer, orig, new)."""
    present = set(number_strings(source)) | set(number_strings(answer))
    cands = [(s, e, t) for s, e, t in find_numbers(answer) if t.isdigit() and len(t) <= 6]
    rng.shuffle(cands)
    for s, e, t in cands:
        n = int(t)
        options = [n + 1, n * 2, n * 10, n + 5] + ([n - 1] if n > 1 else []) + ([n // 2] if n > 3 else [])
        options = [str(o) for o in options if str(o) not in present and o != n]
        if options:
            new = rng.choice(options)
            return answer[:s] + new + answer[e:], t, new
    return None


def perturb_number_swap(answer, source, rng):
    """One number replaced by a DIFFERENT number that does occur in the source
    (attached to another claim) -> (new_answer, orig, new)."""
    src_nums = list(dict.fromkeys(number_strings(source)))
    cands = find_numbers(answer)
    rng.shuffle(cands)
    for s, e, t in cands:
        options = [n for n in src_nums if n != t]
        if options:
            new = rng.choice(options)
            return answer[:s] + new + answer[e:], t, new
    return None


def replace_focus(text, focus, new):
    """Replace the focus term, matching its case style (lower-case in, lower-case out)."""
    m = re.search(r"(?<!\w)" + re.escape(focus) + r"(?!\w)", text, flags=re.IGNORECASE)
    if not m:
        return None
    if new != "\0" and m.group().islower():
        new = new.lower()
    return text[:m.start()] + new + text[m.end():], m.group()


def perturb_entity(answer, focus, other_focuses, source, rng):
    """Focus term replaced by another document's focus term -> (new_answer, orig, new)."""
    pool = [f for f in other_focuses
            if f.lower() != focus.lower() and f.lower() not in source.lower() and len(f) >= 4]
    r = replace_focus(answer, focus, "\0")
    if not pool or not r:
        return None
    new = rng.choice(pool)
    if r[1].islower():
        new = new.lower()
    return r[0].replace("\0", new), r[1], new


def make_hard_supported(answer, lang):
    """Numbers 1-10 written as words (not before % and not in decimals)."""
    words = NUMBER_WORDS.get(lang)
    if not words:
        return None
    out, last, changed = [], 0, False
    for s, e, t in find_numbers(answer):
        if t.isdigit() and 1 <= int(t) <= 10 and answer[e:e + 1] != "%":
            out.append(answer[last:s] + words[int(t) - 1])
            last, changed = e, True
    if not changed:
        return None
    out.append(answer[last:])
    return "".join(out)


def make_referral(answer, lang):
    return answer.rstrip() + " " + REFERRAL[lang]


def make_omission(answer, lang, min_keep=5, min_drop=3):
    """First sentence, or first comma clause; never cut before a condition word."""
    stop = CONDITION_WORDS.get(lang, set())
    sents = split_sentences(answer)
    if len(sents) >= 2 and word_count(sents[0]) >= min_keep:
        rest = " ".join(sents[1:]).split()
        if rest and rest[0].lower().strip(",") not in stop:
            return sents[0]
    k = answer.find(", ")
    if k > 0:
        left, right = answer[:k], answer[k + 2:]
        first = right.split()[0].lower() if right.split() else ""
        opener = left.split()[0].lower() if left.split() else ""
        if (word_count(left) >= min_keep and word_count(right) >= min_drop and first not in stop
                and opener not in OPENER_WORDS.get(lang, set()) and ":" not in left):
            return left.rstrip(".") + "."
    return None


def make_added_fact(i, answer, source, other_answers, focuses, rng):
    """Append the first sentence of another topic's answer -> (new_answer, j, hard)."""
    cands = []
    for j, a in enumerate(other_answers):
        if j == i or not a or focuses[j].lower() == focuses[i].lower():
            continue
        sent = split_sentences(a)[0]
        if sent and sent not in source and sent not in answer and content_overlap(sent, source) < MAX_OVERLAP:
            nums = set(number_strings(sent))
            cands.append((j, sent, bool(nums) and nums <= set(number_strings(source))))
    if not cands:
        return None
    hard = [c for c in cands if c[2]]
    j, sent, is_hard = rng.choice(hard or cands)
    return answer.rstrip() + " " + sent, j, is_hard


def pick_wrong_source(base_idx, answer, sources, focuses, rng):
    """Another topic's passage; prefers one containing all the answer's numbers -> (j, hard)."""
    nums = set(number_strings(answer))
    others = [j for j in range(len(sources))
              if j != base_idx and focuses[j].lower() != focuses[base_idx].lower()
              and focuses[base_idx].lower() not in sources[j].lower()
              and content_overlap(answer, sources[j]) < MAX_OVERLAP]
    if not others:
        return None
    hard = [j for j in others if nums and nums <= set(number_strings(sources[j]))]
    return (rng.choice(hard), True) if hard else (rng.choice(others), False)


# --------------------------------------------------------------------------
# 3. Gold explanations
# --------------------------------------------------------------------------
PASS_REASON = {
    "supported": "None. Every claim in the answer matches the source.",
    "hard_supported": "None. The answer writes some numbers as words, but every value matches the source.",
    "omission": "None. The answer leaves out some details, but everything it states matches the source.",
    "referral": "None. The answer adds advice to see a health worker, which is not a factual claim; "
                "every factual claim matches the source.",
}


def severity_for(item_type, gloss, meta):
    if item_type in PASS_TYPES:
        return "NONE"
    if item_type in ("entity", "wrong_source"):
        return "CRITICAL"
    if item_type in ("number", "number_swap"):
        return "CRITICAL" if is_critical_number(gloss, meta["new"]) else "MINOR"
    if item_type == "added_fact":
        added = meta["added"]
        return "CRITICAL" if any(is_critical_number(added, n) for n in number_strings(added)) else "MINOR"
    raise ValueError(item_type)


def discrepancy_for(item_type, gloss, meta):
    if item_type in PASS_TYPES:
        return PASS_REASON[item_type]
    if item_type == "number":
        return (f'The answer says "{meta["ctx_new"]}", but the source says "{meta["ctx_orig"]}": '
                f"{meta['orig']} was changed to {meta['new']}.")
    if item_type == "number_swap":
        return (f'The answer says "{meta["ctx_new"]}", but the source says "{meta["ctx_orig"]}": '
                f"{meta['orig']} was changed to {meta['new']}, a number the source uses for a different claim.")
    if item_type == "entity":
        return f"The answer is about {meta['new']}, but the source is about {meta['orig']}."
    if item_type == "added_fact":
        return f'The answer adds "{meta["added"]}", which the source does not state.'
    if item_type == "wrong_source":
        return f"The source is about {meta['other']}, not {meta['topic']}; nothing in it supports this claim."
    raise ValueError(item_type)


# --------------------------------------------------------------------------
# 4. Item assembly
# --------------------------------------------------------------------------
def make_items(bases, lang, source_lang, seed=0):
    """Items for one (answer language, source language) condition.
    bases need .answer['en'], .answer[lang] and, if source_lang != 'en',
    .passage_tr[source_lang]. Returns (items, counts)."""
    rng = random.Random(f"{seed}-{lang}-{source_lang}")
    usable = [b for b in bases if "en" in b.answer and lang in b.answer
              and (source_lang == "en" or source_lang in b.passage_tr)]
    all_focus = sorted({b.focus for b in bases})
    counts = {t: 0 for t in ITEM_TYPES}
    counts.update({f"skip_{t}": 0 for t in ITEM_TYPES})
    counts.update({"hard_wrong_source": 0, "hard_added_fact": 0, "critical": 0})
    items = []

    for split in ("train", "val", "test"):
        group = [b for b in usable if b.split == split]
        sources = [b.source(source_lang) for b in group]
        focuses = [b.focus for b in group]
        answers = [b.answer[lang] for b in group]
        answers_en = [b.answer["en"] for b in group]

        for i, b in enumerate(group):
            ans, ans_en, src = answers[i], answers_en[i], sources[i]
            evid = b.evidence(source_lang)

            def add(t, answer, gloss, source=src, evidence=evid, meta=None, hard=False):
                meta = meta or {}
                label = "supported" if t in PASS_TYPES else "unsupported"
                err = {"number": "number", "number_swap": "number", "entity": "entity",
                       "wrong_source": "wrong_source", "added_fact": "other"}.get(t, "none")
                sev = severity_for(t, gloss, meta)
                items.append(Item(
                    id=f"{b.doc_id}-{lang}-{source_lang}-{t}", doc_id=b.doc_id, split=split, lang=lang,
                    source_lang=source_lang, item_type=t, label=label, question=b.question, source=source,
                    answer=answer, error_type=err, severity=sev, gloss_en=gloss, evidence=evidence,
                    discrepancy=discrepancy_for(t, gloss, meta), meta=meta, hard=hard))
                counts[t] += 1
                counts["critical"] += sev == "CRITICAL"

            add("supported", ans, ans_en)

            hs = make_hard_supported(ans, lang)
            if hs:
                add("hard_supported", hs, make_hard_supported(ans_en, "en") or ans_en)
            else:
                counts["skip_hard_supported"] += 1

            om = make_omission(ans, lang)
            if om:
                add("omission", om, make_omission(ans_en, "en") or ans_en)
            else:
                counts["skip_omission"] += 1

            add("referral", make_referral(ans, lang), make_referral(ans_en, "en"))

            for t, fn in (("number", perturb_number), ("number_swap", perturb_number_swap)):
                p = fn(ans, src, rng)
                g = _replace_number(ans_en, p[1], p[2]) if p else None
                if p and g:
                    add(t, p[0], g[0], meta={"orig": p[1], "new": p[2], "ctx_orig": g[1], "ctx_new": g[2]})
                else:
                    counts[f"skip_{t}"] += 1

            pe = perturb_entity(ans, b.focus, all_focus, src, rng)
            ge = replace_focus(ans_en, b.focus, pe[2]) if pe else None
            if pe and ge:
                add("entity", pe[0], ge[0], meta={"orig": pe[1], "new": pe[2]})
            else:
                counts["skip_entity"] += 1

            ws = pick_wrong_source(i, ans, sources, focuses, rng)
            if ws:
                j, hard = ws
                add("wrong_source", ans, ans_en, source=sources[j], evidence=NO_EVIDENCE,
                    meta={"other": focuses[j], "topic": b.focus}, hard=hard)
                counts["hard_wrong_source"] += hard
            else:
                counts["skip_wrong_source"] += 1

            af = make_added_fact(i, ans, src, answers, focuses, rng)
            if af:
                new, j, hard = af
                added_en = split_sentences(answers_en[j])[0]
                add("added_fact", new, ans_en.rstrip() + " " + added_en, meta={"added": added_en}, hard=hard)
                counts["hard_added_fact"] += hard
            else:
                counts["skip_added_fact"] += 1

    return items, counts


def robustness_variants(items):
    """Test-split copies with label-preserving surface changes:
    no_hooks (Hausa typed without hooked letters) and pcm_spelling."""
    out = []
    for it in items:
        if it.split != "test" or it.variant != "original":
            continue
        if it.lang == "ha":
            new, name = strip_hooks(it.answer), "no_hooks"
        elif it.lang == "pcm":
            new, name = pidgin_respell(it.answer), "pcm_spelling"
        else:
            continue
        if new != it.answer:
            d = asdict(it)
            d.update(id=f"{it.id}-{name}", answer=new, variant=name)
            out.append(Item(**d))
    return out


def select_items(items, split, target_n=None, train_langs=("en", "pcm", "ha"), seed=0):
    """Original (non-variant) items of `split` in train_langs. With target_n,
    English is kept in full and each target language is capped (H3)."""
    rows = [i for i in items if i["split"] == split and i["lang"] in train_langs
            and i.get("variant", "original") == "original"]
    if target_n is None:
        return rows
    rng = random.Random(seed)
    out = [i for i in rows if i["lang"] == "en"]
    for lang in sorted({i["lang"] for i in rows} - {"en"}):
        pool = [i for i in rows if i["lang"] == lang]
        out += rng.sample(pool, min(target_n, len(pool)))
    return out


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write((it.to_json() if isinstance(it, Item) else json.dumps(it, ensure_ascii=False)) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
