# Build Plan

Each phase ends with something committed and a result you can quote.

| Phase | What | Where | Output | Status |
|---|---|---|---|---|
| 0 | Protocol, pipeline with gold reasoning traces, judges, metrics, 24 tests | this repo | `PROTOCOL.md`, offline dry run | ✅ done |
| 1 | Build the dataset (Gemini rewrite, sentence-aligned translation, filter) | notebook 01 | `data/items.jsonl`, `data/stats.json` | ☐ |
| 1b | Your checks: translation spot-check (~30–40 per language); Pidgin and Hausa wording in `judge/text.py` (number words, referral sentences, condition and opener words, `PIDGIN_VARIANTS`); read ~10 gold traces | `human_labels/` | % meaning preserved | ☐ |
| 2 | Tokenizer fertility; zero-shot judges: shortcut, Tiny Aya Earth, Qwen3-4B, Llama 3.2 3B (4-bit), Gemini | notebook 02 | H1, H4; model comparison; choice of base | ☐ |
| 3 | QLoRA fine-tune the chosen base; select the epoch on val; test once | notebook 03 | H2 | ☐ |
| 4 | Ablations: verdict-only vs reasoning; data curve (en-only, +100, +500); QLoRA vs 16-bit LoRA | notebook 03 | H3; what the reasoning buys; memory and time table | ☐ |
| 5 | Natural errors: RAGTruth QA (English); ~50 real answers per language in Pidgin and Hausa, checked by you | `scripts/ragtruth_items.py`, `human_labels/` | transfer to real hallucinations | ☐ |
| 6 | Figures, findings, limitations; release the adapter on the Hugging Face Hub | README | finished technical report | ☐ |

## Datasets

| Dataset | Licence | Used for |
|---|---|---|
| MedQuAD (`lavita/MedQuAD`) | CC BY 4.0 | source passages and questions |
| Constructed set (from MedQuAD) | CC BY 4.0 (derived) | training (train), epoch selection (val), evaluation (test) |
| RAGTruth (`wandb/RAGTruth-processed`), QA test | MIT | evaluation only: natural errors in English |
| Human-checked natural answers | ours | evaluation only: natural errors in Pidgin and Hausa |

## Models

| Model | Why | Licence / access |
|---|---|---|
| `CohereLabs/tiny-aya-earth` (3.35B) | Africa-focused; Hausa supported | CC-BY-NC, gated |
| `Qwen/Qwen3-4B` | broadly multilingual, permissive | Apache 2.0 |
| `meta-llama/Llama-3.2-3B-Instruct` | English-centric comparison | Llama 3.2 Community Licence, gated |
| Gemini Flash | API judge and translator | API terms |

Before phase 2: accept the Tiny Aya and Llama licences on Hugging Face, and add `HF_TOKEN` to Colab Secrets.

## Resume milestones

- **After phase 2:** "Built a cross-lingual faithfulness benchmark (English sources, Pidgin and Hausa answers) with known labels and reasoning traces, adversarial controls and robustness sets; benchmarked three small open models and an API judge."
- **After phase 3:** add the fine-tuned result: F1 on Pidgin/Hausa, critical-error recall, cost on one T4.
- **After phase 6:** link the report and the released adapter
