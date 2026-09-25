> Historical local audit, before GitHub publication. Current packaging, license scope and fresh-environment checks: [../README.md](../README.md), [DATA_AND_LICENSES.md](../DATA_AND_LICENSES.md), [VALIDATION.md](../VALIDATION.md).

# Data, weights and distribution scope

The reproducibility bundle is a local research archive. It has not been published, uploaded or submitted.
License files found beside third-party code/weights are preserved; their existence does not automatically cover
all underlying training records or create an unrestricted dataset redistribution right.

| Asset | Provenance | Local evidence |
|---|---|---|
| AMP-Diffusion code/weights | starter kit pinned in `docs/sources.json` | starter kit MIT license; original checkpoint hash |
| ProtGPT2 base | `nferruz/ProtGPT2`, revision in `protein_lm/artifacts/base_manifest.json` | base MIT license and SHA-256 manifest |
| ProtGPT2 adapter | locally trained LoRA | adapter weights, configuration and training data reports |
| GRU GFlowNet | locally trained model | source, weights and reward metadata |
| ESM2-8M | checkpoint identified in original run manifest | model hash; upstream attribution must accompany distribution |
| Antibacterial reference | competition template | original reference hash and source repository |
| AMP/negative training data | distributed starter-kit datasets | data manifests and phase provenance reports |
| Historical MIC data | GRAMPA 2018 snapshot, DBAASP subset | 2,260 derived sequences; original transformation code preserved |
| New completion code | this project | MIT license applies to `completion/src` and new tests/docs only |

The MIC source snapshot has no root license established by the current records. Database terms for constituent data
remain unresolved for public redistribution. A local archive is not a declaration that a public competition submission
meets all licensing conditions. Source and asset hashes identify exactly what was used without expanding those rights.

The existing `phase1/uv.lock` and `protein_lm/uv.lock` preserve model environments. Evaluation-only additions have a
pinned requirements file and are installed separately. No API keys, environment secrets, `.git` internals or virtual
environments belong in the release archive.
