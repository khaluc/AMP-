# AMP ensemble submission

Reproducible computational AMP candidate design with AMP-Diffusion, ProtGPT2-LoRA, and a GRU GFlowNet, followed by frozen MIC prediction and diverse portfolio selection. This repository contains the trained local weights, inference code, exact training splits, ranking documentation, and the frozen 50,000-member library and ranked top 100.

The outputs are computational candidates; antimicrobial activity and safety have not been experimentally established.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone this public repository:

```bash
git clone https://github.com/khaluc/AMP-.git
cd AMP-
uv run generate
```

Python is pinned to **3.10.21**. The root launcher and both model environments have committed `uv.lock` files. No command arguments are required. The default seed is **42**, with frozen branch seeds documented in [METHODS.md](METHODS.md).

Default generation requires an NVIDIA CUDA GPU supporting BF16; the original run used an RTX 4050 Laptop GPU with 6 GiB VRAM. Allow approximately 30 minutes per full inference run on that GPU, plus first-run downloads and installation. First use downloads the pinned 3.13 GB ProtGPT2 base and 133 MB diffusion checkpoint from their original publishers and verifies SHA-256 before loading. All locally trained weights are committed here; upstream download locations and hashes are in [assets/downloads.json](assets/downloads.json).

The command creates:

```text
generate/library.fasta  # 50,000 unique sequences
generate/top.fasta      # 100 ranked candidates
generate/top.csv        # ranking details
generate/run.json       # verification status and output hashes
```

Running the same command again recomputes the outputs in a separate replay directory and compares both FASTA files byte for byte. Cross-hardware identity is not guaranteed; inference rejects a mismatch against the frozen release rather than silently publishing different candidates.

For a faster CPU check using the existing sequence pools:

```bash
uv run generate --mode cached --output generate-cached
```

This explicitly reconstructs saved candidates and reruns scoring/selection checks; it does **not** claim model inference. A CUDA-enabled PyTorch environment is installed even in cached mode because the locked research environments are preserved.

If Windows Application Control prevents execution of the generated `generate.exe` launcher, use the same Python entry point:

```powershell
uv run python -m amp_completion.submission
```

This machine-specific limitation and exact clean-environment checks are recorded in [VALIDATION.md](VALIDATION.md).

## Submission and method

- [Method abstract, models, filtering and ranked-selection protocol](METHODS.md)
- [Frozen library](submission/library.fasta), [top 100 FASTA](submission/top.fasta), [top 100 table](submission/top.csv)
- [Data disclosure, licenses and historical MIC review](DATA_AND_LICENSES.md)
- [Source and weight notices](THIRD_PARTY_NOTICES.md)
- [Validation evidence and remaining eligibility limits](VALIDATION.md)

The top-100 order is the greedy portfolio construction order, not an individually sorted potency ranking. The MIC model uses a historical **E. coli** endpoint; this is not evidence of broad-spectrum or clinical efficacy. Test interval coverage was **276/358 = 77.095%**, below the nominal 90%. No threshold was retuned on test data. No MD or experimental validation is claimed.

`phase1/` contains generation/data preparation, `protein_lm/` the LoRA model, `phase2/` MIC prediction and ranking, `phase3/` portfolio selection, and `completion/` the earlier audit bundle. Original source files remain unchanged so the frozen provenance checks continue to work.

## License and access

Project-owned code is under [MIT](LICENSE). Third-party code, weights and data retain their own licenses and attribution requirements; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Public access gives the organizers `RasmusML` and `szymczakpau` read access without granting write permissions. Eligibility and co-authorship are determined by the competition organizers.
