# Validation of the public repository

Validated on 2026-09-26 on Windows with Python 3.10.21, CUDA PyTorch 2.5.1 and an RTX 4050 Laptop GPU (6 GiB).

| Check | Result | Evidence |
|---|---|---|
| Root launcher | `uv run generate --help` passed | New root environment |
| Default full inference | `uv run --locked generate` passed in 1841.6 seconds | [full.json](validation/full.json), [full-audit.json](validation/full-audit.json) |
| Generator pools | All three newly inferred pools match the original sequences in order | Full audit |
| Library and top 100 | Reconstructed and matched the frozen FASTA hashes | Full audit |
| Repeated cached runs | Two invocations produced byte-identical FASTA outputs | [cached-repeat.json](validation/cached-repeat.json) |
| Public GitHub clone | Clone with no pre-existing virtual environments installed and ran the cached pipeline successfully | [github-clone-check.json](validation/github-clone-check.json) |
| Research tests from public clone | 28/28 passed | Clone check |
| New asset loader tests | 3/3 passed | [entrypoint-tests.xml](validation/entrypoint-tests.xml) |
| Public read access | Available to RasmusML and szymczakpau without write invitations | [github-read-access.json](validation/github-read-access.json) |
| Data source review | Six original input files disclosed; four independently matched to the MIT-licensed HydrAMP data archive | [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) |

The full default run created fresh model environments; it did not reuse the research workspace's virtual environments. The two large upstream model downloads passed their size and SHA-256 checks before inference. The public-clone check is a second clean checkout and fresh set of dependency environments, and runs the explicitly labeled cached mode. It does not count as a second fresh full-inference run.

The full default run reproduces the original model-generated pools and release outputs; the earlier full checkpoint replay also matched those pools. The current public launcher was run once in full mode and twice in cached mode. Same-host reproducibility is established by the frozen-output comparisons; cross-hardware identity is not claimed. No Windows security policy was changed.

Library SHA-256: `36b147ebe114d5176305f1b9fa59ce8f2f8fef7799c493c377f6ebb17a191aa9`. Top-100 SHA-256: `c38400c4d492ba0115a94183c28b9101c911f77ab8928ccbb182e0da1a9cd729`.

The original organizer sequence-validation functions passed for the frozen library and top 100. The official clone-and-generate-twice validator was not rerun end to end in this session; the equivalent individual sequence checks and actual full/cached runs are identified above.

The MIC model's test interval coverage remains 276/358 = 77.095%, below nominal 90%. Pretraining overlap and a heterogeneous historical E. coli endpoint limit biological interpretation. No MD, wet-lab efficacy, toxicity or clinical validation is claimed. These are scientific limits rather than changes made by publication. Co-authorship and acceptance remain the organizers' decisions.
