"""Tests for data construction, gold explanations, prompt/parser, baseline and
metrics. Run on real MedQuAD text (fixture). No GPU or API needed."""

import json
import os
import random
import subprocess
import sys

import pytest

from judge.baselines import shortcut_predict
from judge.construct import (ITEM_TYPES, PASS_TYPES, make_added_fact, make_hard_supported, make_items,
                             make_omission, make_referral, perturb_entity, perturb_number, perturb_number_swap,
                             pick_wrong_source, robustness_variants, select_items, select_passages,
                             severity_for, split_by_document)
from judge.metrics import coverage_at_precision, critical_metrics, ece, explanation_metrics, paired_bootstrap_diff, prf
from judge.prompt import build_messages, parse_output, target_text
from judge.text import (clean_medquad, content_overlap, find_numbers, is_critical_number, normalize_text,
                        number_phrase, pidgin_respell, same_numbers, split_sentences, strip_hooks)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures", "medquad_sample.jsonl")


def records():
    with open(FIX) as f:
        return [json.loads(l) for l in f]


def bases_with_fake_translation(n=60):
    """Identity 'translation': exercises every code path without an API."""
    bases = split_by_document(select_passages(records(), n_docs=n))
    for b in bases:
        b.answer = {"en": b.sentence, "pcm": b.sentence, "ha": b.sentence}
        b.passage_tr = {"pcm": list(b.passage_sents), "ha": list(b.passage_sents)}
    return bases


# ---------------------------------------------------------------- text
def test_numbers_found_and_compared():
    t = "Adults need 7 to 9 hours; 1,000 people; 5.7 percent; COVID-19; H1N1 virus."
    assert [n for _, _, n in find_numbers(t)] == ["7", "9", "1,000", "5.7", "19"]
    assert same_numbers("take 2 then 3", "3 and 2 later") and not same_numbers("take 2", "take 20")


def test_clean_and_split_medquad():
    text = "Summary : A1C is a test. Results:    - Normal is below 5.7 percent    - High is above 6.5 percent"
    sents = split_sentences(clean_medquad(text))
    assert sents[0] == "A1C is a test." and any("6.5" in s for s in sents)


def test_normalisation_and_robustness_transforms():
    assert normalize_text("Ma’aikaci  ʼya") == "Ma'aikaci 'ya"
    assert strip_hooks("ɗaya ƙwaya ɓarna") == "daya kwaya barna"
    assert pidgin_respell("Wetin dey happen?") == "Watin de happen?"


def test_critical_number_rules():
    assert is_critical_number("Normal A1C is below 6.4 percent.", "6.4")
    assert is_critical_number("Take 2 tablets daily.", "2")
    assert is_critical_number("Children ages 5 and older", "5")
    assert not is_critical_number("About 25 percent of adults", "25")
    assert not is_critical_number("It affects 3 million people.", "3")
    assert number_phrase("below 6.4 percent.", "6.4") == "6.4 percent"
    from judge.text import context_window
    t = "People at risk are over 60, have had a stroke."
    k = t.index("60")
    assert context_window(t, k, k + 2) == "risk are over 60, have had a"
    assert number_phrase("between 50 and 60 years", "50") == "50"          # 'and' is not a unit
    assert is_critical_number("between 20 and 50 years of age", "50")      # age threshold
    assert not is_critical_number("25 percent of adults", "25")
    assert is_critical_number("more likely to develop type 7 diabetes", "7")   # condition subtype


# ---------------------------------------------------------------- constructions
def test_select_passages_and_document_splits():
    bases = split_by_document(select_passages(records(), n_docs=60))
    assert len(bases) > 20 and len({b.doc_id for b in bases}) == len(bases)
    assert {b.split for b in bases} == {"train", "val", "test"}
    for b in bases:
        assert find_numbers(b.sentence) and b.sentence == b.passage_sents[b.sent_idx]


def test_number_perturbations():
    rng = random.Random(0)
    new, orig, val = perturb_number("Check A1C 2 times a year.", "Test at least 2 times a year.", rng)
    assert orig == "2" and val not in ("2",) and val in new
    src = "A normal level is below 5.7 percent. Diabetes is above 6.5 percent."
    new, orig, val = perturb_number_swap("Diabetes means above 6.5 percent.", src, rng)
    assert (orig, val) == ("6.5", "5.7") and "5.7" in new
    assert perturb_number_swap("Above 6.5 percent.", "Only 6.5 here.", rng) is None


def test_supported_variants():
    assert make_hard_supported("Take 2 tablets for 3 days.", "en") == "Take two tablets for three days."
    assert make_hard_supported("Sha 2 a rana.", "ha") == "Sha biyu a rana."
    assert make_hard_supported("Rates rose 5% in 2020.", "en") is None
    assert make_referral("Normal A1C dey below 5.7 percent.", "pcm").endswith("make dem advise you.")
    assert make_omission("A1C is a blood test. It measures sugar over 3 months.", "en") == "A1C is a blood test."
    assert make_omission("Take 2 tablets each day, unless you are pregnant.", "en") is None
    # never keep only a leading clause: that would drop the main claim
    assert make_omission("During the first 3 months of pregnancy, it might cause a birth defect.", "en") is None
    assert make_omission("There are three types: Type 1, the most common form, causes pain.", "en") is None


def test_entity_and_foreign_content():
    rng = random.Random(0)
    new, orig, val = perturb_entity("Asthma affects 25 million people.", "asthma", ["Diabetes", "Asthma"], "Asthma", rng)
    assert new == "Diabetes affects 25 million people." and (orig, val) == ("Asthma", "Diabetes")
    new, _, val = perturb_entity("The risk of colorectal cancer rises.", "colorectal cancer", ["Fever"], "x", rng)
    assert new == "The risk of fever rises." and val == "fever"
    sources = ["About 30 percent of adults have asthma.", "Flu lasts 2 weeks.", "Up to 30 percent of children get measles."]
    assert pick_wrong_source(0, "About 30 percent of adults have asthma.", sources, ["asthma", "flu", "measles"], rng) == (2, True)
    src = "Prediabetes means blood glucose is higher than normal. An A1C blood test checks for prediabetes."
    others = ["Prediabetes raises risk.", "A1C is a blood test for prediabetes.", "Measles is 90 percent contagious."]
    assert content_overlap(others[1], src) >= 0.5
    for _ in range(10):  # the sentence restating the source is never used
        new, j, _ = make_added_fact(0, others[0], src, others, ["prediabetes", "a1c", "measles"], rng)
        assert j == 2


# ---------------------------------------------------------------- items and gold explanations
def test_items_have_correct_labels_and_gold_explanations():
    bases = bases_with_fake_translation()
    items, counts = make_items(bases, "pcm", "en")
    assert len({i.id for i in items}) == len(items)
    assert {"supported", "referral", "number", "number_swap", "wrong_source", "added_fact"} <= {i.item_type for i in items}
    for i in items:
        assert i.label == ("supported" if i.item_type in PASS_TYPES else "unsupported")
        assert (i.severity == "NONE") == (i.label == "supported")
        if i.item_type == "wrong_source":
            assert i.evidence == "none found"
        else:
            assert i.evidence in i.source                      # gold evidence is quoted from the source
        if i.item_type in ("number", "number_swap"):
            assert i.meta["new"] in i.answer and i.meta["new"] in i.gloss_en
            assert i.meta["new"] in i.discrepancy and i.meta["orig"] in i.discrepancy
        if i.item_type == "number_swap":
            assert set(n for _, _, n in find_numbers(i.answer)) <= set(n for _, _, n in find_numbers(i.source))
        if i.item_type == "number":
            assert i.meta["new"] not in [n for _, _, n in find_numbers(i.source)]
    by_doc = {}
    for i in items:
        by_doc.setdefault(i.doc_id, set()).add(i.split)
    assert all(len(v) == 1 for v in by_doc.values())          # no document in two splits


def test_same_language_evidence_is_from_translated_source():
    bases = bases_with_fake_translation()
    for b in bases:  # make the 'translation' distinguishable
        b.passage_tr["ha"] = [s + " [ha]" for s in b.passage_sents]
    items, _ = make_items(bases, "ha", "ha")
    for i in items:
        if i.item_type != "wrong_source":
            assert i.evidence.endswith("[ha]") and i.evidence in i.source


def test_target_text_parses_back_exactly():
    items, _ = make_items(bases_with_fake_translation(), "ha", "en")
    for it in [i for i in items][:60]:
        d = it.__dict__
        p = parse_output(target_text(d))
        assert p["complete"]
        assert p["verdict"] == it.label and p["error_type"] == it.error_type and p["severity"] == it.severity
        assert p["claim_en"] == it.gloss_en and p["discrepancy"] == it.discrepancy
        assert p["evidence"] == it.evidence
    assert target_text(items[0].__dict__, "verdict") in ("Verdict: PASS", "Verdict: FAIL")


def test_severity_rules():
    assert severity_for("supported", "x", {}) == "NONE"
    assert severity_for("entity", "x", {}) == "CRITICAL"
    assert severity_for("number", "Normal is below 6.4 percent.", {"new": "6.4"}) == "CRITICAL"
    assert severity_for("number", "It affects 3 million people.", {"new": "3"}) == "MINOR"


def test_robustness_variants_keep_labels():
    bases = bases_with_fake_translation()
    for b in bases:
        b.answer["ha"] = "Ƙwayar cuta: " + b.answer["ha"]
        b.answer["pcm"] = "Wetin dey: " + b.answer["pcm"]
    items = make_items(bases, "ha", "en")[0] + make_items(bases, "pcm", "en")[0]
    vs = robustness_variants(items)
    assert vs and all(v.split == "test" for v in vs)
    assert {v.variant for v in vs} == {"no_hooks", "pcm_spelling"}
    orig = {i.id: i for i in items}
    for v in vs:
        o = orig[v.id.rsplit("-", 1)[0]]
        assert v.label == o.label and v.answer != o.answer


def test_select_items_for_training():
    items = [{"split": "train", "lang": l, "id": f"{l}{k}", "variant": "original"} for l in ("en", "pcm", "ha") for k in range(50)]
    items += [{"split": "train", "lang": "ha", "id": "x", "variant": "no_hooks"}]
    assert len(select_items(items, "train")) == 150            # variants never used for training
    capped = select_items(items, "train", target_n=10)
    assert sum(i["lang"] == "en" for i in capped) == 50 and sum(i["lang"] == "ha" for i in capped) == 10


def test_build_script_dry_run(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_dataset.py"), "--dry_run",
                        "--medquad_jsonl", FIX, "--n_docs", "40", "--out", str(tmp_path), "--n_spotcheck", "5"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    stats = json.load(open(tmp_path / "stats.json"))
    assert stats["n_items"] > 0 and stats["n_common_passages"] > 0
    assert (tmp_path / "spotcheck_ha.csv").exists()


# ---------------------------------------------------------------- prompt and parser
def test_prompt_contains_rubric_and_format():
    m = build_messages({"source": "S", "question": "Q", "answer": "A"})
    assert "Rubric:" in m[0]["content"] and "1. Claim (in English)" in m[0]["content"]
    assert "SOURCE:\nS" in m[1]["content"]
    assert "Verdict: <PASS or FAIL>" in build_messages({"source": "S", "question": "Q", "answer": "A"}, "verdict")[0]["content"]


def test_parse_output_variants():
    good = ('1. Claim (in English): Normal is below 6.4 percent.\n2. Source evidence: "Normal is below 5.7 percent."\n'
            '3. Discrepancy: The answer says 6.4 percent, but the source says 5.7.\n4. Error type: number\n'
            '5. Severity: CRITICAL\nVerdict: FAIL')
    p = parse_output(good)
    assert p["complete"] and p["verdict"] == "unsupported" and p["evidence"] == "Normal is below 5.7 percent."
    p = parse_output("Some rambling... **Verdict: pass**")
    assert p["verdict"] == "supported" and not p["complete"]
    assert parse_output("I think it is fine")["verdict"] is None


# ---------------------------------------------------------------- baseline
def test_shortcut_baseline_failure_modes_by_design():
    src = "Take 2 tablets daily for 5 days."
    assert shortcut_predict({"source": src, "answer": "Take 2 tablets daily."})["verdict"] == "supported"
    assert shortcut_predict({"source": src, "answer": "Take 20 tablets daily."})["verdict"] == "unsupported"
    assert shortcut_predict({"source": src, "answer": "Take two tablets daily."})["verdict"] == "unsupported"  # wrong
    assert shortcut_predict({"source": src, "answer": "Take 5 tablets daily."})["verdict"] == "supported"     # wrong


# ---------------------------------------------------------------- metrics
def _row(i, label, pred, p=None, **kw):
    return {"id": str(i), "label": label, "pred": pred, "p_supported": p, **kw}


def test_prf_counts_parse_failures_as_wrong():
    rows = [_row(0, "unsupported", "unsupported"), _row(1, "unsupported", None),
            _row(2, "supported", "supported"), _row(3, "supported", None)]
    m = prf(rows)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["parse_fail"] == 0.5


def test_ece_and_coverage():
    rows = [_row(i, "supported", "supported", 0.9) for i in range(9)] + [_row(9, "unsupported", "supported", 0.9)]
    assert ece(rows) == pytest.approx(0.0, abs=1e-9)
    rows2 = [_row(i, "supported", "supported", 0.99) for i in range(18)] + \
            [_row(18 + k, "unsupported", "unsupported", 0.1) for k in range(2)]
    cov = coverage_at_precision(rows2)
    assert cov["threshold"] == 0.99 and cov["coverage"] == pytest.approx(0.9)


def test_paired_bootstrap():
    rows = [_row(i, "unsupported" if i % 2 else "supported", "unsupported") for i in range(40)]
    assert paired_bootstrap_diff(rows, rows, n_boot=50) == (0, 0, 0)
    # B gets 10 FAIL items wrong that A gets right: a real, paired difference
    worse = [{**r, "pred": "supported"} if r["label"] == "unsupported" and int(r["id"]) < 20 else r for r in rows]
    point, lo, hi = paired_bootstrap_diff(rows, worse, stat="accuracy", n_boot=300)
    assert point == pytest.approx(0.25) and 0 < lo <= point <= hi


def test_limit_samples_whole_documents():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_judge import pick_items
    items = [{"id": f"{d}-{l}-{t}", "doc_id": str(d), "split": "test", "lang": l, "variant": "original"}
             for d in range(30) for l in ("en", "ha") for t in range(3)]
    picked = pick_items(items, "test", 30)
    docs = {i["doc_id"] for i in picked}
    assert all(sum(i["doc_id"] == d for i in picked) == 6 for d in docs)   # every condition of each document


def test_explanation_and_critical_metrics():
    src = "Normal is below 5.7 percent. Prediabetes is 5.7 to 6.4 percent."
    base = dict(source=src, evidence="Normal is below 5.7 percent.", gloss_en="Normal is below 6.4 percent.",
                item_type="number_swap", meta={"orig": "5.7", "new": "6.4"}, severity="CRITICAL")
    good = _row(0, "unsupported", "unsupported", **base, pred_generated="unsupported", format_complete=True,
                pred_error_type="number", pred_severity="CRITICAL", pred_claim_en="Normal is below 6.4 percent.",
                pred_evidence="Normal is below 5.7 percent.",
                pred_discrepancy="The answer says 6.4 percent, but the source says 5.7.")
    bad = {**good, "id": "1", "pred_evidence": "Normal is below 4 percent.", "pred_discrepancy": "Wrong number.",
           "pred": "supported", "pred_generated": "supported", "pred_severity": None}
    e = explanation_metrics([good, bad])
    assert e["evidence_valid"] == 0.5 and e["evidence_correct"] == 0.5 and e["names_values"] == 1.0
    assert e["claim_chrf"] == pytest.approx(100.0)
    c = critical_metrics([good, bad])
    assert c["critical_recall"] == 0.5 and c["severity_agreement"] == 1.0


def test_fertility_and_ragtruth_helpers():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from ragtruth_items import convert
    from tokenizer_fertility import fertility
    assert fertility(lambda t: 2 * len(t.split()), ["a b c", "d e"]) == 2.0
    row = {"id": 7, "query": "Q?", "context": "C.", "output": "Answer with error.",
           "hallucination_labels": '[{"start": 12, "end": 17, "text": "error"}]'}
    assert convert(row)["label"] == "unsupported" and convert({**row, "hallucination_labels": "[]"})["label"] == "supported"
