# Data card

**Contents.** `items.jsonl` holds the label-by-construction evaluation items. Each item includes its gold reasoning trace (English claim, quoted evidence, discrepancy, error type, severity), plus test-only robustness copies (`variant`). `bases.json` holds the passages, answers and translations they were built from. `stats.json` holds the counts and filter/skip rates.

**Source.** MedQuAD (Ben Abacha & Demner-Fushman, 2019), Hugging Face copy `lavita/MedQuAD`, licensed CC BY 4.0. Passages are consumer health text from NIH websites.

**Processing.** An LLM (Gemini; model recorded in `stats.json`) rewrote one passage sentence per document as an assistant answer, then translated passages (sentence by sentence, kept aligned) and answers into Nigerian Pidgin and Hausa. All text is NFC-normalised. Translations were back-translated and kept only if chrF ≥ 50 and every number survived. Errors were then injected by rule (see PROTOCOL.md §5.3), so every label is known by construction.

**Intended use.** Evaluating whether judges detect unsupported content in health answers. This is **not** a source of medical advice: answers may be deliberately wrong.

**Known issues.** The translations are machine-made; see `human_labels/` for the spot-check results.
