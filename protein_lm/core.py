from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "models/protgpt2"
OUT = HERE / "artifacts"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "phase1"))
from common import AA, fasta, valid, write_fasta, sha, dump, csv_sequences, split
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.pad_token = tok.eos_token
    return tok


def prefix(tok, label="AMP"):
    # Existing vocabulary only; control prefix is never part of output peptide.
    return [tok.eos_token_id] + tok.encode(label + "\n", add_special_tokens=False)


def encode(tok, sequence, label):
    p = prefix(tok, label)
    body = tok.encode(sequence, add_special_tokens=False) + [tok.eos_token_id]
    return p + body, [-100] * len(p) + body


def batch(examples, tok, device="cuda"):
    size = max(len(x[0]) for x in examples)
    ids = torch.full((len(examples), size), tok.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, (x, y) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x, device=device)
        labels[i, :len(y)] = torch.tensor(y, device=device)
        attention[i, :len(x)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def load_model(adapter=None, trainable=False):
    # BF16 base + FP32 LoRA parameters: no base optimizer states, no remote code.
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["c_attn", "c_proj"], fan_in_fan_out=True, bias="none"))
    model.to("cuda")
    model.config.pad_token_id = 0
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model.eval()
        model.config.use_cache = True
    return model
