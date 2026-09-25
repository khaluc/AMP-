# Data attribution

- **HydrAMP published dataset**: Szymczak et al. (2022), version 1.2.0, [DOI 10.5281/zenodo.7420278](https://zenodo.org/records/7420278), MIT as recorded in its dataset metadata. The existing [University of Warsaw MIT notice](../starter-kits/hydramp/LICENSE) is retained. Four input files match the published data archive exactly; see `docs/hydramp-dataset-license.json`.
- **AMP-Diffusion training compilation**: Torres et al., Cell Biomaterials (2025), [DOI 10.1016/j.celbio.2025.100183](https://doi.org/10.1016/j.celbio.2025.100183). The official data release specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source databases are DRAMP, APD and DBAASP. The original training FASTA is included unchanged.
- **DBAASP**: database and contributors, [current redistribution/acknowledgement terms](https://dbaasp.org/terms-and-conditions); Pirtskhalava et al., [DBAASP v3](https://doi.org/10.1093/nar/gkaa991). The original DBAASP exclusion FASTA and a filtered historical E. coli measurement subset are included. Filtering and per-sequence median aggregation are documented in `phase2/prepare.py`.
- **UniProt Consortium**: [CC BY 4.0](https://www.uniprot.org/help/license). The original UniProt-derived HydrAMP input and locally prepared sequence partitions retain this attribution.
- **APD, DRAMP, dbAMP and AMP Scanner**: original database authors and curators are acknowledged in `DATA_AND_LICENSES.md` and the publication/source dataset notices.

Local adaptations comprise canonical-residue/length filtering, duplicate removal, exact-sequence exclusions, deterministic partitioning, and MIC aggregation. These changes are not the work of the original data authors. No endorsement is implied.
