# Generator 2: ProtGPT2 with LoRA

This replaces the earlier locally trained 415k-parameter PeptideLM with the author's
pretrained **nferruz/ProtGPT2**, pinned to revision
`ff981fc96771d6f6d3b6453f94772d3cb6c7b90b`. The original baseline remains in `phase1/`.
ProtGPT2 is a causal protein language model. We do not reinterpret ESM-2's masked-LM
checkpoint as an autoregressive AMP-GPT.

The model card describes 738M parameters; the downloaded GPT-2 configuration actually
contains **774,030,080 base parameters**. LoRA adds **8,110,080 trainable parameters**.
There is no substitution with a randomly initialized small model.

## Training

- LoRA rank 16, alpha 32, dropout 0.05; GPT-2 `c_attn` and `c_proj` modules, including
  attention and MLP output projections; `fan_in_fan_out=True` for GPT-2 Conv1D weights.
- Frozen BF16 base; FP32 trainable adapters; gradient checkpointing; AdamW 2e-4.
- Three epochs, microbatch 64, gradient accumulation 4 (effective 256 examples).
- 15,117 AMP positives + 15,117 assumed-negative UniProt short sequences. Validation:
  1,642 positives + 1,642 controls. Distinct existing-vocabulary control prefixes
  `AMP\n` / `NONAMP\n`; prefix and padding labels are masked, real EOS remains supervised.
- Best adapter selected on validation AMP token NLL. This is a language-model development
  metric, not antimicrobial potency or a measured hit rate.
- `--smoke --batch-size 64` tests two forward/backward optimizer updates on the longest
  tokenized examples. Do not infer training fit merely from the model loading successfully.

The negative source is `Uniprot_0_25_train.csv`; `unlabelled_negative.csv` mostly contains
long proteins and had only four valid 8–50-aa entries. The short UniProt controls are
not verified inactive peptides. Their <=25-aa length range can confound the control label.
The model is not an activity classifier and label-conditional generation is not a potency guarantee.

Exact MIC reservations from Phase 1 are excluded from new fine-tuning data. The validation
split is not homology/lab independent. Unknown overlap with ProtGPT2's original UniRef50
pretraining remains; these data do not establish a conformal guarantee.

## Generation

Positive `AMP` control prefix, top-k 100, top-p 0.95, temperature 1, seed 3042 plus round
number. BPE tokens may contain multiple residues: masking enforces 8–50 **amino acids**,
not 8–50 BPE tokens. Token decoding removes whitespace and permits only standard residues;
pure whitespace/other tokens are forbidden. No slicing of long peptides or random fallback.
The pool excludes the official reference and all exact fine-tuning/validation sequences.
Every complete batch is saved with checkpoint/data/code hashes for resume.
The production configuration uses batch 128 (the conservative CLI default is 32).
Strict deterministic algorithms remain enabled: the top-p cumulative sum runs on
CPU because PyTorch 2.5 does not provide a deterministic CUDA float cumsum.

`assemble.py` writes a separate `artifacts/library_v2/` with 16,667 diffusion + 16,667
ProtGPT2-LoRA + 16,666 existing GFlowNet candidates. It applies global exact deduplication
and known-data exclusions. The old GFlowNet's reward teacher remains the small PeptideLM;
upgrading that reward is a separate task. No top-100 or conformal selection is performed here.

## Run from the AMP root (PowerShell)

```powershell
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.cache\python"
uv sync --project protein_lm --locked
protein_lm/.venv/Scripts/python.exe protein_lm/download.py
protein_lm/.venv/Scripts/python.exe protein_lm/prepare.py
protein_lm/.venv/Scripts/python.exe protein_lm/train.py --smoke --batch-size 64
protein_lm/.venv/Scripts/python.exe protein_lm/train.py --batch-size 64 --accumulation 4 --epochs 3
# If interrupted, repeat that training command with --resume.
protein_lm/.venv/Scripts/python.exe protein_lm/sample.py --count 17500 --batch-size 128
# If interrupted, repeat that sampling command with --resume.
python protein_lm/assemble.py
protein_lm/.venv/Scripts/python.exe protein_lm/verify.py
protein_lm/.venv/Scripts/python.exe protein_lm/report.py
```

Check the training log and `artifacts/training/complete.json` before claiming training
finished. Resume rejects changed hyperparameters/data. Adapter and optimizer checkpoints
are saved every 20 optimizer steps and at epoch boundaries. The inference pool uses its
own independent resume state. `verify.py` expects a fresh `artifacts/replay/` output.

## Sources

- Author's checkpoint: https://huggingface.co/nferruz/ProtGPT2
- PEFT LoRA: https://huggingface.co/docs/peft/v0.13.0/en/package_reference/lora
- Base files and upstream license text are preserved in `models/protgpt2/`; SHA-256
  verification, including the Hub LFS weight hash, is in `artifacts/base_manifest.json`.
- Both pretraining and fine-tuning source licenses/terms must accompany any publication.
