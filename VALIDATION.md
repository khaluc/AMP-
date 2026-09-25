# Validation of the public repository

This file is updated from actual run evidence before publication is finalized.

## Established before packaging

- 28 research-pipeline tests passed in the original locked model environments.
- Full checkpoint inference reconstructed each saved generator pool in order on the original machine.
- Final FASTA hashes: library `36b147ebe114d5176305f1b9fa59ce8f2f8fef7799c493c377f6ebb17a191aa9`; top 100 `c38400c4d492ba0115a94183c28b9101c911f77ab8928ccbb182e0da1a9cd729`.
- Original organizer sequence-validation functions passed for the 50,000-member library and top 100.
- The original source files, model weights, feature cache, frozen thresholds and seeds are preserved byte for byte.

## Public packaging checks

- `uv lock` succeeded using Python 3.10.21 and a new root environment.
- `uv run generate --help` executed the standard launcher successfully.
- 3 asset-loader tests passed: path escape, changed existing bytes and untrusted origins are rejected.
- Two `uv run --locked generate --mode cached --output generate-cached` invocations passed in the new environments and produced identical library/top FASTA bytes. See `validation/cached.json` and `validation/cached-repeat.json`.
- The two large upstream model downloads completed and passed their pinned size and SHA-256 checks before GPU inference started.
- Full `uv run --locked generate` is being tested in newly created project Python environments; evidence is recorded in `validation/full.json` when it finishes. Do not interpret a missing or nonzero result as a successful clean-environment run.
- This uses the same Windows/GPU host. It is a fresh dependency environment, not a separate physical machine or an independent biological validation.

The older `completion/` report describes a Windows launcher block encountered in an earlier environment. The new root launcher has started successfully; no Windows security policy was changed.

The model's 77.095% test coverage, pretraining overlap, heterogeneous MIC endpoint and lack of experimental validation remain limitations. They are not repaired by packaging or reproducibility checks. Co-authorship is not claimed; see the separate data-license review.
