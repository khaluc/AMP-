# data/

Two distinct things live here; do not confuse them:

## `antibacterial.fasta`: challenge reference set (NOT training data)

The **AMP Challenge 2027 reference set** of known antibacterial peptides, copied verbatim
from the [challenge template](https://github.com/szczurek-lab/amp-challenge-2027). Used only
by the generator/validator to enforce two challenge rules:

- no library sequence may be **identical** to any sequence here;
- no top-100 sequence may exceed **80% Levenshtein similarity** to any sequence here.

## `training/`: HydrAMP's actual training data

The curated dataset HydrAMP was trained on (peptides ≤25 aa), as released with the
publication. Full disclosure for the challenge's training-data requirement.

| File | Content | Source | Rows |
|------|---------|--------|------|
| `mic_data.csv` | Sequences with measured MIC vs *E. coli*; `value` = log₁₀(MIC/µM) | GRAMPA (Witten & Witten) | 4,546 |
| `unlabelled_positive.csv` | Known AMP sequences (positives) | dbAMP, DRAMP, AMP Scanner | 20,313 |
| `unlabelled_negative.csv` | Assumed non-AMP sequences (negatives) | UniProt (filtered, CD-HIT) | 25,762 |
| `Uniprot_0_25_{train,val,test}.csv` | Biological-diversity sequences (225k, split) | UniProt | 180,194 / 22,524 / 22,524 |
| `veltri_{positive,negative}.csv` | AMP-classifier training set | AMP Scanner v2 (Veltri et al.) | 2,021 / 1,897 |
| `dbaasp/*.fasta` | E. coli / S. aureus high- & low-activity sets | DBAASP | 3,038 clean + activity splits |

Sequences ≤25 aa, standard amino acids only; shorter sequences are padded during training.
See Szymczak et al. 2023, *Nat. Commun.* 14, 1453 (Methods → Data collection) and the
HydrAMP repo's `scripts/dataset_preparation.ipynb` for the exact assembly.

**Provenance / terms.** All sources are public and were redistributed with the HydrAMP
release; no proprietary or non-public data was used. This is a copy of that release. Users
should confirm each upstream database's terms of use for their own purposes.
