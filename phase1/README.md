# Phase 1 — Three-generator AMP portfolio

This is an experimental implementation, not a validated therapeutic design method.
It implements Phase 1 only. No conformal, MIC hit-rate, mechanism-of-action, or official
competition-score guarantee is attached to its output.

## Components

1. **AMP-Diffusion:** official downloaded checkpoint and ESM2-8M decoder. Uses the
   original 1,000-training-timestep checkpoint. Both the original ancestral sampler and
   a DDIM eta=0, 100-inference-step adapter are available. The user selected DDIM after
   measuring ~82 seconds per 64 sequences for ancestral inference on the RTX 4050.
   DDIM is a disclosed inference variant; equal biological quality is not established.
2. **Peptide LM:** local 128-wide, two-layer causal Transformer, pretrained on the
   supplied UniProt short-peptide training subset then fine-tuned on AMP data. This is
   not the published AMP-GPT model. Length is an explicit context; actions are 20 amino acids.
3. **GFlowNet:** separate 128-wide GRU policy, supervised AMP warm start followed by
   trajectory-balance training. The append-only tree has deterministic backward transitions.
   A separate learned log partition function is used for each length context.

The GFlowNet's positive terminal reward is:

`exp(log P_LM(sequence | length) - property_penalty - 8 * exact_reference_match)`.

`property_penalty` is the mean squared standardized charge density, hydrophobic fraction,
and adjacent-repeat fraction relative to AMP training data, clipped per descriptor at 9.
This combines a frozen sequence prior, empirical property realism and exact novelty.
It is a **proxy reward**, not an activity/MIC predictor. The LM and GFlowNet share a
teacher-based reward dependency and training sources; they are not statistically independent.
No claim of exact convergence to the target reward distribution is made.

## Data and leakage limits

`artifacts/data/report.json` records source hashes and counts. New-model data excludes
all exact sequences in the supplied MIC CSV and cleaned DBAASP FASTA. Those records
are reserved for later data audit, not automatically a valid conformal calibration set.
The 90/10 sequence-hash split is for model-development monitoring only: it does not
separate homologous families, assay laboratories or publication dates. AMP validation
sequences are excluded from local LM pretraining by exact match.

The published diffusion checkpoint has its own historical training set and may overlap
reserved MIC records. That cannot be undone by filtering our new training data. External
data with appropriate assay metadata and an independent validation design will be needed
before claiming calibration validity. Source datasets' licenses/terms remain applicable.

## Commands (PowerShell, from the AMP root)

```powershell
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.cache\python"
uv sync --project phase1 --locked
python -c "import sys; sys.path.insert(0, 'phase1'); from common import prepare; prepare()"
phase1/.venv/Scripts/python.exe -m pytest phase1/test_phase1.py -q
phase1/.venv/Scripts/python.exe phase1/train.py lm
phase1/.venv/Scripts/python.exe phase1/train.py gflownet
phase1/.venv/Scripts/python.exe phase1/generate.py pool diffusion --count 17500 --batch-size 512 --sampler ddim --diffusion-steps 100 --output phase1/artifacts/pools/diffusion_ddim100.fasta
phase1/.venv/Scripts/python.exe phase1/generate.py pool lm --count 17500 --batch-size 64 --seed 1042
phase1/.venv/Scripts/python.exe phase1/generate.py pool gflownet --count 17500 --batch-size 64 --seed 2042
phase1/.venv/Scripts/python.exe phase1/generate.py merge --diffusion-pool phase1/artifacts/pools/diffusion_ddim100.fasta
phase1/.venv/Scripts/python.exe phase1/audit_library.py
```

Run small generation checks before full pools. GPU work is sequential to fit 6 GB VRAM.
Production generation uses distinct fixed random streams: diffusion=42, LM=1042,
GFlowNet=2042. A first LM/GFlowNet pilot using the same seed 42 had 804 exact overlaps
between its pools; those outputs are preserved under `artifacts/experiments/shared-seed-42/`.
Changing generation seeds does not alter model training or the fixed reward.
For each pool, `--resume` continues only when checkpoint hash, reference hash, seed,
device, batch size and existing FASTA hash match. Do not change batch size mid-run.
An interruption between writing FASTA and metadata fails safely rather than guessing state.
Round seeds make continuation deterministic in the same software/hardware environment.
Cross-platform byte identity is not claimed. Outputs retain whole batches, so pool sizes
can exceed the requested minimum.

The final merge enforces **16,667 diffusion + 16,667 LM + 16,666 GFlowNet = 50,000**,
global sequence uniqueness, canonical alphabet, length 8–50 and zero exact overlap with
the official reference. Insufficient pools fail; no random generator or other branch
silently fills a quota. Extend the affected pool using `--resume` and a larger `--count`.

`artifacts/library/provenance.csv` attributes each sequence to its generator.
`artifacts/library/validation.json` records structural checks and source hashes.
These checks do not establish activity, synthetic accessibility or free-terminal behavior.
Top-100 selection and the 80% similarity requirement belong to later phases.

## Research references

- https://github.com/szczurek-lab/ampdiffusion-starter-kit
- https://www.jmlr.org/papers/v24/22-0364.html
- https://arxiv.org/abs/2201.13259 (trajectory balance)
- https://arxiv.org/abs/2010.02502 (DDIM)

The original proposal's assertions about 22/25 successes, embedding clusters as MoA,
and automatic GFlowNet diversity superiority are hypotheses or unsupported guarantees,
not implemented assumptions.
