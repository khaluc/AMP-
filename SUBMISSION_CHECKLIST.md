# Submission requirements

| Requested item | Status / location |
|---|---|
| Method abstract | `METHODS.md` |
| Library of 50,000 AMP candidates | `submission/library.fasta` |
| Ranked top 100 and selection documentation | `submission/top.fasta`, `submission/top.csv`, `METHODS.md` |
| Training data, external databases and filters | `DATA_AND_LICENSES.md`, `assets/training_sources.json`; original inputs and prepared splits included |
| Public model weights and inference | Local trained weights committed; pinned upstream models downloaded and hash-verified automatically |
| Organizer read access | Public repository; both named accounts have public read access |
| OSI code license and third-party notices | Root MIT, Apache-2.0/MIT model notices, BSD template notice, dataset license/attribution review |
| Python and uv lock | `.python-version`, root `uv.lock`, both model-project lock files |
| Standard default entry point | `uv run generate`; tested full inference in fresh Python environments |
| Fixed default randomness | Seed 42 with frozen branch seeds and byte comparison against original model outputs |
| Clean checkout verification | Public clone installed fresh environments, cached reconstruction and 28 research tests passed |

These records document the requested repository preparation and computational checks. They do not grant co-authorship or substitute for the organizers' acceptance decision. See `VALIDATION.md` for the exact scope of repeated-run checks and the existing scientific limitations.
