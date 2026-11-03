"""
QLoRA fine-tuning of an open Hugging Face model as a groundedness judge.

    # QLoRA (4-bit base + LoRA adapters) on all training data
    python scripts/train_qlora.py --model CohereLabs/tiny-aya-earth --out checkpoints/aya-earth-qlora
    # H3 data curve: all English + 100 items per target language
    python scripts/train_qlora.py --model CohereLabs/tiny-aya-earth --target_n 100 --out checkpoints/aya-earth-t100
    # Ablation: 16-bit LoRA (same everything, base not quantised)
    python scripts/train_qlora.py --model CohereLabs/tiny-aya-earth --no_4bit --out checkpoints/aya-earth-lora16
    # Ablation: verdict only, no reasoning
    python scripts/train_qlora.py --model CohereLabs/tiny-aya-earth --mode verdict --out checkpoints/aya-earth-verdict

The model learns to write the five-step reasoning and the PASS/FAIL verdict
for a (source, question, answer) prompt (--mode verdict: the verdict only, for
the ablation). Loss is computed ONLY on that output, never on the prompt. Adapters are saved after every epoch; pick the best one with
scripts/run_judge.py on the validation split.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge.construct import read_jsonl, select_items  # noqa: E402
from judge.hf_judge import chat_prompt  # noqa: E402
from judge.prompt import target_text  # noqa: E402


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def encode(tok, item, max_len, mode):
    """input_ids = prompt + target; labels = -100 on the prompt, so the loss
    only covers the reasoning and verdict the judge must write."""
    p = tok(chat_prompt(tok, item, mode), add_special_tokens=False)["input_ids"]
    t = tok(target_text(item, mode) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:
        return None  # skipped and counted; never silently truncated
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def make_collator(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for k, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[k, :L] = torch.tensor(b["input_ids"])
            labels[k, :L] = torch.tensor(b["labels"])
            mask[k, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


# --------------------------------------------------------------------------
# 2. Model: 4-bit base + LoRA adapters
# --------------------------------------------------------------------------
def build_model(model_id, use_4bit, r, alpha, dropout):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,                      # store frozen base weights in 4 bits
        bnb_4bit_quant_type="nf4",              # NormalFloat4: suited to normally distributed weights
        bnb_4bit_use_double_quant=True,         # also quantise the quantisation constants
        bnb_4bit_compute_dtype=torch.float16,   # de-quantise to fp16 for each matmul (T4 has no bf16)
    ) if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 torch_dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False              # incompatible with gradient checkpointing
    if use_4bit:
        # Casts norms to fp32, enables input grads and gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=r,                          # rank of the update matrices A (d x r) and B (r x d)
        lora_alpha=alpha,             # update is scaled by alpha / r
        lora_dropout=dropout,
        target_modules="all-linear",  # every linear layer in attention + MLP, as in the QLoRA paper
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # Trainable (LoRA) weights in fp32 so fp16 mixed precision can unscale their gradients
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_4bit", action="store_true", help="16-bit LoRA ablation")
    ap.add_argument("--target_n", type=int, default=None, help="H3: cap per target language")
    ap.add_argument("--train_langs", nargs="+", default=["en", "pcm", "ha"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--mode", default="reasoning", choices=["reasoning", "verdict"])
    ap.add_argument("--max_train", type=int, default=None, help="random cap on training items (quick runs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = read_jsonl(a.items)
    train_items = select_items(items, "train", a.target_n, a.train_langs, a.seed)
    if a.max_train and a.max_train < len(train_items):
        train_items = random.Random(a.seed).sample(train_items, a.max_train)
    val_items = random.Random(a.seed).sample(select_items(items, "val"), min(300, len(select_items(items, "val"))))
    enc = lambda xs: [e for e in (encode(tok, i, a.max_len, a.mode) for i in xs) if e]
    train_ds, val_ds = enc(train_items), enc(val_items)
    print(f"train {len(train_ds)}/{len(train_items)} (skipped {len(train_items) - len(train_ds)} over {a.max_len} tokens), "
          f"val {len(val_ds)}")

    model = build_model(a.model, not a.no_4bit, a.r, a.alpha, a.dropout)
    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,     # effective batch = batch_size * grad_accum
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,                            # QLoRA paper setting
        optim="paged_adamw_8bit" if not a.no_4bit else "adamw_torch",  # paged optimiser absorbs memory spikes
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    stats = {
        "model": a.model, "method": "lora16" if a.no_4bit else "qlora", "mode": a.mode, "target_n": a.target_n,
        "train_items": len(train_ds), "epochs": a.epochs, "r": a.r, "alpha": a.alpha, "lr": a.lr,
        "train_minutes": (time.time() - t0) / 60,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "log_history": trainer.state.log_history,
    }
    trainer.save_model(os.path.join(a.out, "final"))
    json.dump(stats, open(os.path.join(a.out, "train_stats.json"), "w"), indent=1)
    print(f"done in {stats['train_minutes']:.1f} min, peak memory {stats['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
