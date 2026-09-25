# Training data, licenses and distribution review

Reviewed on 2026-09-26. This file describes the public repository; earlier reports in `completion/` record the pre-publication local audit.

## Project license and scope

The root [MIT license](LICENSE) covers the project-owned Python code in `phase1/`, `phase2/`, `phase3/`, `protein_lm/`, `src/`, `completion/` and the associated project documentation. Original third-party license and attribution notices remain applicable. This grant does not relicense upstream databases or third-party model weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Locally trained baseline Transformer/GRU weights, the multiobjective GFlowNet and the random-forest predictor are distributed with the project under MIT. The ProtGPT2 LoRA adapter is distributed under Apache-2.0 with attribution to ProtGPT2, conservatively retaining the model-card terms as well as the upstream MIT license file. The base model's own files are not relabeled.

## Exact training and exclusion data

No proprietary or private dataset was introduced by this project. Public source revisions and SHA-256 values are listed in [assets/training_sources.json](assets/training_sources.json); the actual prepared splits used by the local models are committed:

| Use | Exact local records | Source and handling |
|---|---|---|
| Local baseline LM and GFlowNet positives | `phase1/artifacts/data/amp_train.fasta` (15,117); `amp_val.fasta` (1,642) | AMP-Diffusion training set plus HydrAMP public positives; canonical lengths 8–50; exact MIC exclusions; deterministic sequence split |
| Baseline LM pretraining | `phase1/artifacts/data/uniprot_train.fasta` (162,189); `uniprot_val.fasta` (17,905) | Public UniProt-derived HydrAMP set; exact exclusions; assumed non-AMP status is not experimentally verified |
| LoRA fine-tuning | `protein_lm/artifacts/data/train.json`, `validation.json` | 15,117 AMP plus 15,117 assumed non-AMP training sequences; 1,642 plus 1,642 validation sequences; exact splits and labels included |
| MIC regression | `phase2/artifacts/dataset.csv` | 2,260 unique sequences; DBAASP-only, unmodified E. coli records from the pinned 2018 GRAMPA snapshot; median of already-log10 MIC values; split labels included |
| Original MIC measurements used | `phase2/artifacts/source/eligible_measurements.csv` | Filtered DBAASP records before per-sequence aggregation; source URLs retained |
| Sequence exclusion | `phase1/artifacts/data/reserved_mic.fasta`; `data/reference/antibacterial.fasta` | Exact exclusions; reference data is used for filtering, not model fitting |
| Published diffusion checkpoint | [AMP-Diffusion data disclosure](starter-kits/ampdiffusion/data/README.md) | 19,670 public training sequences; pinned source link in the training-source manifest |
| ProtGPT2 base pretraining | [Public UniRef50 2021_04 dataset](https://huggingface.co/datasets/nferruz/UR50_2021_04) | Pretraining by the original authors, not performed locally; local LoRA changes are disclosed separately |
| ESM2 representation model | [Original ESM repository and model description](https://github.com/facebookresearch/esm) | Published pretrained representation model; checkpoint hashes retained |

`phase1/artifacts/data/amp_provenance.csv` records the contributing sources for each positive sequence. The exact filters and partitions are implemented in `phase1/common.py`, `protein_lm/prepare.py` and `phase2/prepare.py`. Predictor choice used validation data; the final fit combined train and validation. Neither calibration nor test was used to fit the final predictor or retune its interval widths.

No individual peptide was manually edited or promoted. Manual intervention consisted of choosing the modeling approach, published sources, quotas and deterministic filtering/ranking protocol; the code then generated and selected the submitted sequences. The full library uses only canonical letters, length 8–50 and unique sequences. Sequence-only files cannot certify synthesized terminal chemistry; intended synthesis is linear peptides with free termini and no modifications.

## Historical MIC review

The MIC predictor uses only the **DBAASP subset**, not all databases contained in GRAMPA. The canonical [DBAASP terms page](https://dbaasp.org/terms-and-conditions), checked on the review date, permits downloading, adapting and redistributing data with public acknowledgement. We retain DBAASP source URLs and acknowledge the database and its contributors here. See Pirtskhalava et al., [DBAASP v3, Nucleic Acids Research 2021](https://doi.org/10.1093/nar/gkaa991).

An older [DBAASP PDF](https://www.dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf) contains conflicting distribution statements. The current canonical terms page is the basis for distributing the clearly identified DBAASP subset; this review does not claim that the older PDF licenses unrelated databases in GRAMPA.

The pinned [GRAMPA repository](https://github.com/zswitten/Antimicrobial-Peptides/tree/a6819dcd671291af8013627a2d2fbc32858346bc) has no root license established in its snapshot. Therefore this repository **does not redistribute the full mixed-source `grampa.csv` or the upstream GRAMPA Python files**. The source URL, revision, transformation-code hashes and full-snapshot hash remain disclosed. Three small upstream files used by the frozen provenance audit are downloaded directly from their publisher when the entry point runs; they are not imported, executed or covered by this project's MIT grant. The original project-owned `phase2/download.py` can acquire the research source for retraining, but it is not needed for frozen inference and its latest-branch download is not a replacement for the pinned disclosure.

## Other source terms and attribution

- **UniProt**: [CC BY 4.0 terms](https://www.uniprot.org/help/license). UniProt Consortium data are used through the pinned HydrAMP distribution; the prepared local sequence splits are adaptations.
- **DRAMP**: [database attribution/license information](https://dramp.cpu-bioinfor.org/). Credit the DRAMP database and the published AMP-Diffusion/HydrAMP datasets.
- **APD**: [official FAQ](https://aps.unmc.edu/faq) permits use of downloads with acknowledgement. Credit Wang, Li and Wang, APD3, Nucleic Acids Research 2016, and the APD website.
- **AMP-Diffusion**: Torres et al., [Cell Biomaterials 2025](https://doi.org/10.1016/j.celbio.2025.100183). The official starter kit identifies its published training compilation as CC BY 4.0; preserve that attribution and the database acknowledgements.
- **HydrAMP data / AMP Scanner / dbAMP inputs**: Szymczak et al., [Nature Communications 2023](https://doi.org/10.1038/s41467-023-36994-z), the [pinned HydrAMP starter-kit data disclosure](starter-kits/hydramp/data/README.md), and its public upstream sources. The original aggregate source files are linked rather than copied into this repository. Prepared local splits are identified explicitly as adaptations.

The starter-kit MIT software licenses do not by themselves prove unrestricted relicensing of every historical database entry. In particular, the aggregate AMP-positive/exclusion sets include records with several upstream origins. This submission discloses those origins and their notices rather than claiming all training data is MIT. **A blanket permissive license for every historical upstream database record is not established by this review.** Organizer acceptance of the mixed-source disclosures remains an eligibility question; publication and computational checks alone do not certify co-authorship eligibility.

## Downloaded and included weights

All locally trained weights are present in the repository. The two large, unchanged upstream models are downloaded at pinned revisions and verified against [assets/downloads.json](assets/downloads.json). This avoids committing multi-gigabyte base weights to ordinary Git. No private model access, credentials or paid API is needed. CUDA hardware and software dependencies are still required for full inference.
