"""
Open-model judge (Hugging Face transformers), used zero-shot and after QLoRA.

1. The model writes its five-step reasoning (greedy decoding).
2. The verdict probability is then read by **constrained scoring**: given the
   prompt plus the model's own reasoning up to "Verdict:", we compare the
   log-probabilities of " PASS" and " FAIL". This gives p_supported (used for
   thresholds and calibration) and a valid verdict even when a small model
   breaks the format. The verdict the model actually wrote is recorded
   separately (pred_generated), along with whether the format was complete.

In verdict-only mode (ablation) step 1 is skipped and the prefix is just
"Verdict:".

Base models load in 4-bit NF4 by default: the same setting the QLoRA judge is
trained and served in, so zero-shot vs fine-tuned isolates fine-tuning.
"""

import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompt import build_messages, parse_output


def load_model(model_id, load_4bit=True, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16) if load_4bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto",
                                                 quantization_config=quant)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def chat_prompt(tok, item, mode="reasoning"):
    """Model's chat template. Templates that reject a system role get the
    system text prepended to the user turn. enable_thinking=False turns off
    'thinking' output in models that have it (e.g. Qwen3); others ignore it."""
    msgs = build_messages(item, mode)
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        return tok.apply_chat_template(msgs, **kw)
    except Exception:
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        return tok.apply_chat_template(merged, **kw)


@torch.no_grad()
def _continuation_logprob(model, tok, prefix_ids, continuation):
    """Sum of log-probs of `continuation` after `prefix_ids` (tokenised separately)."""
    c_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([prefix_ids, c_ids], dim=1).to(model.device)
    logp = torch.log_softmax(model(full).logits[0, :-1].float(), -1)
    start = prefix_ids.shape[1] - 1
    return logp[start:].gather(1, full[0, 1:][start:, None]).sum().item()


@torch.no_grad()
def judge_item(model, tok, item, mode="reasoning", max_new_tokens=320):
    t0 = time.time()
    prompt = chat_prompt(tok, item, mode)
    text = ""
    if mode == "reasoning":
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        cut = text.find("Verdict:")
        prefix = text[:cut] + "Verdict:" if cut >= 0 else text.rstrip() + "\nVerdict:"
    else:
        prefix = "Verdict:"
    prefix_ids = tok(prompt + prefix, return_tensors="pt", add_special_tokens=False).input_ids
    lp_pass = _continuation_logprob(model, tok, prefix_ids, " PASS")
    lp_fail = _continuation_logprob(model, tok, prefix_ids, " FAIL")
    p_sup = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, lp_fail - lp_pass))))
    parsed = parse_output(text) if mode == "reasoning" else {}
    return {
        "pred": "supported" if p_sup >= 0.5 else "unsupported",
        "p_supported": p_sup,
        "pred_generated": parsed.get("verdict"),
        "format_complete": parsed.get("complete"),
        "pred_error_type": parsed.get("error_type"),
        "pred_severity": parsed.get("severity"),
        "pred_claim_en": parsed.get("claim_en"),
        "pred_evidence": parsed.get("evidence"),
        "pred_discrepancy": parsed.get("discrepancy"),
        "raw": text,
        "latency_s": time.time() - t0,
        "mode": mode,
    }
