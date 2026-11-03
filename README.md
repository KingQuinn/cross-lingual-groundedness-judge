# Cross-Lingual Groundedness Judge

**Can a small, 4-bit model check whether a health answer in Nigerian Pidgin or Hausa is faithful to its English source, and explain why not?**

Khadijah Yunusa · *Work in progress: dataset pipeline, judges and evaluation framework complete; experiments running*

---

## Summary

Health information in Nigeria increasingly reaches people through assistants that answer in Nigerian Pidgin and Hausa, while the trusted documents behind them are written in English. If an assistant changes a dosage threshold or invents a claim while translating, the error is both dangerous and hard to catch: fluent reviewers are scarce, and large API judges are costly and less reliable in low-resource languages.

This project builds and evaluates a **cross-lingual faithfulness judge**. Given an English source, a question, and an answer in Pidgin or Hausa, it returns **PASS or FAIL** after a five-step, plain-language reasoning trace:
1. it restates the claim in English;
2. it quotes the source evidence;
3. it states exactly what differs;
4. it names the error type;
5. it rates the severity.

The judge is a small open model fine-tuned with **4-bit QLoRA**, so it can be trained and run on a single low-cost GPU.

The evaluation is built so that **every label and every gold explanation is known by construction**:
- errors are injected into translated MedQuAD answers;
- adversarial cases defeat string matching;
- copies of the test set mimic how people actually type (Hausa without hooked letters, Pidgin spelling variants);
- natural-error test sets measure how far the results carry over to real hallucinations.

The judge's explanations are scored automatically: Is the quoted evidence really in the source? Did it understand the Pidgin or Hausa claim? Did it name both the wrong and the correct value?

The method, hypotheses and decision rules are fixed in advance in **[PROTOCOL.md](PROTOCOL.md)**.

## Example

```
SOURCE (English):  A normal A1C level is below 5.7 percent. Prediabetes is between 5.7 to 6.4 percent.
QUESTION:          Wetin be normal A1C level?
ANSWER (Pidgin):   Normal A1C level dey below 6.4 percent.

1. Claim (in English): A normal A1C level is below 6.4 percent.
2. Source evidence: "A normal A1C level is below 5.7 percent."
3. Discrepancy: The answer says "normal A1C level is below 6.4 percent", but the source says "normal A1C
   level is below 5.7 percent": 5.7 was changed to 6.4, a number the source uses for a different claim.
4. Error type: number
5. Severity: CRITICAL
Verdict: FAIL
```

## Hypotheses

| | Hypothesis |
|---|---|
| **H1** | Small zero-shot judges catch unfaithful answers less well in Pidgin and Hausa than in English; the gap is larger for Hausa. |
| **H2** | 4-bit QLoRA fine-tuning narrows the gap and brings a small judge close to a large API judge. |
| **H3** | English-only training transfers partly; a little target-language data recovers most of the gap. |
| **H4** | Checking against an English source is harder than checking against a same-language source. |

**Exploratory:**
- Africa-focused vs multilingual vs English-centric base models (Tiny Aya Earth, Qwen3-4B, Llama 3.2 3B), alongside tokenizer fertility.
- What the reasoning trace adds over a bare verdict.
- QLoRA vs 16-bit LoRA.

## Method

```
MedQuAD passage (English) ──► English answer (LLM rewrite; numbers checked)
        │                              │
        └── translated sentence by sentence (Pidgin, Hausa) ── back-translation + chrF + number filter
                                       │
                                       ▼
   items with known labels AND known reasoning traces, for 5 conditions (primary: English source → Pidgin/Hausa answer)
   PASS: supported · hard_supported · omission · referral
   FAIL: number · number_swap · entity · wrong_source · added_fact
   + robustness copies (Hausa without hooked letters, Pidgin spelling variants)
                                       │
                                       ▼
   judges: shortcut · Tiny Aya Earth / Qwen3-4B / Llama 3.2 3B (4-bit, zero-shot) · Gemini · QLoRA judge
                                       │
                                       ▼
   F1 (FAIL class) with paired bootstrap CIs · recall on CRITICAL errors · explanation-quality checks
   · robustness · calibration · latency, memory, tokenizer fertility · auto-approval rate at 95% precision
```

## Results

<!-- RESULTS:START -->
*Results appear here after running the notebooks; `scripts/analyze.py` writes this section.*
<!-- RESULTS:END -->

<!-- TODO after results: findings for H1 and H4 (zero-shot), then H2 and H3 (fine-tuned), the model comparison, and what the explanations show. -->

## Deployment view

| Judge output | Action |
|---|---|
| PASS, confident | send to the user |
| FAIL, MINOR | human review |
| FAIL, CRITICAL | block, and show a safe fallback ("please speak to a health worker") |

The results report the share of answers that can be auto-approved at 95% precision, recall on CRITICAL errors, latency, and training memory and time on a single T4.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 24 tests on real MedQuAD text; no GPU or API needed
python scripts/build_dataset.py --dry_run --medquad_jsonl tests/fixtures/medquad_sample.jsonl   # offline pipeline check
```

The experiments run on Colab:
1. `notebooks/01_build_data.ipynb`: the dataset (Gemini API);
2. `notebooks/02_zero_shot_judges.ipynb`: the zero-shot judges and tokenizer analysis (T4);
3. `notebooks/03_qlora_finetune.ipynb`: fine-tuning and ablations (T4).

## Repository

```
PROTOCOL.md      research questions, rubric, hypotheses, data, metrics, decision rules (fixed in advance)
BUILD_PLAN.md    phases, datasets, models, milestones
judge/
  text.py        normalisation, numbers, severity rules, robustness transforms, language word lists
  construct.py   passages, document splits, constructions, gold reasoning traces, robustness copies
  llm_stage.py   batched and cached Gemini calls: rewrite, sentence-aligned translation, back-translation filter
  prompt.py      rubric, reasoning format, fine-tuning target, parser
  hf_judge.py    open-model judge: reasoning generation + scored verdict, 4-bit
  baselines.py   string-matching shortcut
  metrics.py     F1, paired bootstrap, ECE, coverage, explanation quality, critical recall
scripts/         build_dataset · run_judge · train_qlora · analyze · tokenizer_fertility · ragtruth_items
notebooks/       01 build data · 02 zero-shot judges · 03 QLoRA fine-tuning
human_labels/    translation spot-checks and the natural-error set
tests/           unit tests (fixture: 200 MedlinePlus Q&A pairs from MedQuAD, CC BY 4.0)
```

## Limitations

- Constructed errors contain one error each and are simpler than real hallucinations; the natural-error sets measure that difference.
- Pidgin and Hausa text is machine-translated, so the filter and spot-checks reduce translation errors but don't remove them.
- The sources are US health websites with few dosages, so CRITICAL errors are mostly thresholds, ages, durations and condition names.
- Severity is rule-based and approximate.
- There is a single annotator.
- The judge checks **faithfulness only**. A faithful answer can still be incomplete or unsafe for a particular patient.
- Written reasoning is an auditable trace, not proof of how the model decided.

## References

- Ben Abacha & Demner-Fushman (2019). *A question-entailment approach to question answering.* BMC Bioinformatics. (MedQuAD)
- Niu et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* ACL.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Cohere Labs (2026). *Tiny Aya.* arXiv:2603.11510.
- Turpin et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS.
- *Multi-Hall-SA: A Cross-lingual Benchmark for Multi-Type Hallucination Detection in Low-Resource South African Languages.* Findings of EACL 2026.
- *Evaluating Retrieval-Augmented Generation for Medication Question Answering on Nigerian Drug Labels in Yorùbá.* LoResLM 2026.
- *Challenges and Recommendations for LLM-as-a-Judge in Multilingual Settings and for Low-Resource Languages.* arXiv:2607.02235 (2026).
- *Understanding Language Bias in LLM-as-a-Judge Systems.* arXiv:2601.13649 (2026).
