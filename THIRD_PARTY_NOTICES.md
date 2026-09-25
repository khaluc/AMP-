# Third-party notices

The root MIT grant applies to project-owned contributions. Preserve the following upstream notices when redistributing covered assets.

| Component | License / terms | Included notice |
|---|---|---|
| AMP Challenge template and validator | BSD-3-Clause; use the exact upstream text | `competition-template/LICENSE` |
| AMP-Diffusion starter-kit code and released weights | MIT | `starter-kits/ampdiffusion/LICENSE` |
| HydrAMP starter-kit documentation/code | MIT | `starter-kits/hydramp/LICENSE` |
| ESM code and pretrained ESM2 checkpoint | MIT, Meta Platforms, Inc. and affiliates | `LICENSES/ESM-MIT.txt` |
| ProtGPT2 base model | Model-card metadata says Apache-2.0; upstream also supplies an MIT LICENSE file | Both `LICENSES/Apache-2.0.txt` and `models/protgpt2/LICENSE`; original `models/protgpt2/README.md` retained |
| Locally trained ProtGPT2 LoRA adapter | Apache-2.0 for the distributed adaptation | `LICENSES/Apache-2.0.txt`; local training procedure and changes in `METHODS.md` |
| Curated peptide/UniProt/DBAASP training records | Source-specific attribution and terms | `DATA_AND_LICENSES.md`, pinned data README files, source manifest and per-sequence provenance |

ProtGPT2 is by Noelia Ferruz and collaborators. The local adaptation adds LoRA rank-16 updates in the attention/projection modules and AMP/NONAMP conditioning using existing tokenizer tokens; it does not replace or claim authorship of the base model. The model card's Apache designation and its MIT file are both retained rather than silently resolving the discrepancy as MIT-only.

AMP-Diffusion, ESM2, ProtGPT2 and the datasets remain the work of their original authors. This repository is an independent exploratory submission and is not endorsed by those authors or the competition organizers.

The three unlicensed GRAMPA provenance scripts and the full mixed-source raw CSV are not redistributed here. Their public upstream URLs and cryptographic hashes are disclosed, and the audit can retrieve the original scripts directly. See `DATA_AND_LICENSES.md` for the remaining mixed-data licensing limitation.
