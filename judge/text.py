"""Text utilities: sentence splitting, number finding, number words."""

import re

# Standalone numbers: "12", "1,000", "3.5". Not part of a word ("H1N1"), but a
# number after a hyphen ("COVID-19") is included; changing it is still an error.
NUMBER_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")

# Number words 1-10 used for hard_supported items. Pidgin uses English number
# words. Hausa list: checked by a fluent reader (see PROTOCOL.md section 5.3).
NUMBER_WORDS = {
    "en": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "pcm": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "ha": ["ɗaya", "biyu", "uku", "huɗu", "biyar", "shida", "bakwai", "takwas", "tara", "goma"],
}

# Fixed referral sentence appended for `referral` items (supported by rubric
# rule 4). Pidgin and Hausa wording: checked by a fluent reader.
REFERRAL = {
    "en": "Please talk to a health worker for advice.",
    "pcm": "Abeg talk to health worker make dem advise you.",
    "ha": "Don Allah ku tuntuɓi ma'aikacin lafiya don samun shawara.",
}

# Words that start a condition or exception. Dropping a clause that starts with
# one of these could change the meaning ("take daily, unless pregnant"), so
# omission items never cut there.
# Words that open a subordinate or prepositional clause. An omission never
# keeps only such a clause, which would cut away the main claim
# ("During the first 3 months of pregnancy, it might cause ...").
OPENER_WORDS = {
    "en": {"during", "when", "if", "after", "before", "because", "although", "while", "since", "in", "for",
           "as", "unless", "until", "once", "by", "with", "without", "among", "of", "to", "from", "at", "on"},
    "pcm": {"during", "when", "if", "after", "before", "because", "as", "since", "for", "for inside", "wen"},
    "ha": {"idan", "lokacin", "bayan", "kafin", "saboda", "da", "a", "cikin", "tun", "yayin"},
}

CONDITION_WORDS = {
    "en": {"unless", "except", "if", "only", "when", "until", "provided", "but", "however"},
    "pcm": {"unless", "except", "if", "only", "when", "until", "but", "sake"},
    "ha": {"idan", "in", "sai", "amma", "har", "kawai"},
}

_ABBREV = ("e.g.", "i.e.", "U.S.", "etc.", "Dr.", "vs.", "approx.", "No.")


def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_medquad(text):
    """Remove MedQuAD artefacts: 'Summary :' prefixes and bullet markers,
    turning each bullet into its own sentence."""
    text = re.sub(r"^\s*Summary\s*:\s*", "", text)
    text = re.sub(r"\s+-\s+(?=[A-Z0-9])", ". ", text)       # bullets -> sentence breaks
    text = re.sub(r"\.\s*\.", ".", text)                    # '..' created above
    text = re.sub(r":\s*\.\s", ": ", text)                  # 'list:. Item' -> 'list: Item'
    text = re.sub(r"([^.!?:])\s*\.\s(?=[A-Z0-9])", r"\1. ", text)
    return normalize_ws(text)


def split_sentences(text):
    """Simple rule-based splitter: break after . ! ? followed by space + capital/digit,
    unless the token is a known abbreviation."""
    text = normalize_ws(text)
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z0-9\"(])", text):
        end = m.end()
        prev_token = text[start:end].split()[-1] if text[start:end].split() else ""
        if prev_token in _ABBREV:
            continue
        out.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if s]


def find_numbers(text):
    """List of (start, end, string) for every standalone number."""
    return [(m.start(), m.end(), m.group()) for m in NUMBER_RE.finditer(text)]


def number_strings(text):
    return [s for _, _, s in find_numbers(text)]


def same_numbers(a, b):
    """True if a and b contain exactly the same multiset of numbers."""
    return sorted(number_strings(a)) == sorted(number_strings(b))


def number_word_tokens(text, lang):
    words = set(NUMBER_WORDS.get(lang, []))
    return [w for w in re.findall(r"[\wɗƙɓƴ']+", text.lower()) if w in words]


def word_count(text):
    return len(text.split())


_STOP = {"that", "this", "with", "from", "have", "your", "they", "their", "there", "which", "about", "also",
         "than", "more", "most", "some", "many", "when", "what", "will", "been", "were", "into", "other",
         "dey", "wey", "una", "make", "dem", "for", "the", "and", "kuma", "wanda", "cikin", "suna", "domin"}


def content_overlap(text, reference):
    """Share of `text`'s content words (4+ letters, not stopwords) that also
    occur in `reference`. Used to keep 'unsupported' constructions from
    accidentally restating something the source already says."""
    words = [w for w in re.findall(r"[\wɗƙɓƴ']+", text.lower()) if len(w) >= 4 and w not in _STOP]
    if not words:
        return 0.0
    ref = set(re.findall(r"[\wɗƙɓƴ']+", reference.lower()))
    return sum(w in ref for w in words) / len(words)


# --------------------------------------------------------------------------
# Normalisation and robustness transforms
# --------------------------------------------------------------------------
import unicodedata  # noqa: E402

_APOSTROPHES = "’‘ʼʻ`´"


def normalize_text(text):
    """NFC Unicode form (one encoding for ɓ, ɗ, ƙ ...) and a single apostrophe
    character, so the same word always tokenises the same way."""
    text = unicodedata.normalize("NFC", text)
    for a in _APOSTROPHES:
        text = text.replace(a, "'")
    return normalize_ws(text)


_HOOKS = str.maketrans({"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ƴ": "Y"})


def strip_hooks(text):
    """Hausa as often typed on phones: hooked letters replaced by plain ones."""
    return text.translate(_HOOKS)


# Common alternative Pidgin spellings. Edited by a fluent reader; every
# substitution must keep the meaning unchanged.
PIDGIN_VARIANTS = {
    "dey": "de",
    "wetin": "watin",
    "pikin": "pickin",
    "make": "mek",
    "abeg": "abeg",   # placeholder: identity, edit or extend
}


def pidgin_respell(text):
    def sub(m):
        w = m.group(0)
        v = PIDGIN_VARIANTS.get(w.lower())
        if not v:
            return w
        return v.capitalize() if w[0].isupper() else v
    return re.sub(r"[A-Za-z]+", sub, text)


# --------------------------------------------------------------------------
# Severity (PROTOCOL.md section 2.2): rule-based, computed on the English gloss
# --------------------------------------------------------------------------
CRITICAL_UNITS = {
    "mg", "mcg", "g", "ml", "milligrams", "micrograms", "grams", "tablet", "tablets", "pill", "pills",
    "capsule", "capsules", "dose", "doses", "drop", "drops", "times", "hour", "hours", "minute", "minutes",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "kg", "pounds", "lb", "lbs",
    "mmhg", "mg/dl", "degrees", "f", "c", "percent", "%",
}
_POPULATION = {"of", "people", "adults", "children", "americans", "million", "billion", "cases", "deaths", "women", "men"}


_UNIT_WORDS = CRITICAL_UNITS | {"million", "billion", "thousand", "people", "adults", "children", "cases",
                                 "deaths", "women", "men", "americans"}


def number_phrase(text, value):
    """The number plus its unit, if the next word is a unit (e.g. '6.4 percent',
    '2 tablets'); otherwise the number alone. For readable explanations."""
    for s, e, t in find_numbers(text):
        if t == value:
            m = re.match(r"\s*(%|[A-Za-z/]+)", text[e:])
            if m and m.group(1).lower() in _UNIT_WORDS:
                return t + ("" if m.group(1) == "%" else " " + m.group(1))
            return t
    return value


def is_critical_number(text, value):
    """True if `value` in `text` is a dose, frequency, duration, age or clinical
    threshold: followed by a clinical unit and not by a population word."""
    for s, e, t in find_numbers(text):
        if t != value:
            continue
        after = re.findall(r"%|[A-Za-z/]+", text[e:e + 30].lower())
        before = re.findall(r"[a-z]+", text[max(0, s - 12):s].lower())
        if before and before[-1] in {"age", "ages", "aged", "type", "stage", "grade", "phase"}:
            return True   # age threshold, or a condition subtype/stage ('type 2 diabetes')
        if not after:
            return False
        unit, nxt = after[0], (after[1] if len(after) > 1 else "")
        if unit not in CRITICAL_UNITS:
            continue
        if unit in ("percent", "%") and nxt == "of":   # '25 percent of adults' = population statistic
            continue
        if nxt in _POPULATION - {"of"}:
            continue
        return True
    return False


def context_window(text, start, end, words=3):
    """The changed span with up to `words` words either side, cut directly
    from the text (punctuation and spacing preserved), for explanations."""
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    before = [t for t in toks if t[1] <= start][-words:]
    after = [t for t in toks if t[0] >= end][:words]
    a = before[0][0] if before else start
    b = after[-1][1] if after else end
    return text[a:b].strip(" ,;.")
