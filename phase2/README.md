# Phase 2: experimental MIC regression and conformal evaluation

Run with the locked `phase1` environment. `uv sync --project phase1 --locked`.
The 6 GB GPU runs one GPU task at a time; no parallel model training is required.

## Data and endpoint

Pinned GRAMPA snapshot `a6819dcd671291af8013627a2d2fbc32858346bc`, originally collected
in 2018, restricted to DBAASP-source E. coli assays with known modification metadata,
`is_modified=False`, `modifications=[]`, standard uppercase residues, length 8–50.
This is **not a current direct DBAASP API download**. Source URL/hash, raw CSV and
upstream conversion code are preserved in `artifacts/source/`.
Upstream `src/load_data.py` converts MIC to uM then log10. Do not log a second time.
The response is the median log10 MIC for each sequence across available assays/strains.
It does not describe any particular competition strain. Censoring flags and complete
assay conditions were not retained by this snapshot. Modified/unknown peptides are
excluded, not assumed comparable to free-terminus candidates. Source annotation is
not a chemical verification of the structure.

Sequence connected components at >80% normalized Levenshtein edit similarity are
assigned to train/validation/calibration/test by fixed seeds 42/43/44, approximately
60/10/15/15 of groups. Actual sequence counts differ with group sizes. No component
crosses splits. This does not imply independence by publication/lab or between groups.
Audit generator-data overlap separately: published diffusion training has substantial
overlap with held-out MIC sequences. Local LM/GFlowNet data have zero exact overlap.

## Frozen predictor and intervals

Mean residue embedding from frozen ESM2-8M layer 6 (320 dimensions), plus ten documented
sequence descriptors. Ridge(10/100), ExtraTrees and RandomForest are compared by
validation MAE. The chosen model is refitted on train+validation only. Calibration
and test labels do not choose its hyperparameters or the GFlowNet reward weights.

With n calibration sequences, alpha=0.1, take order statistic
`ceil((n+1)*(1-alpha))` of absolute residuals, or infinity if that rank exceeds n.
Signed residuals also yield a separate one-sided upper-bound diagnostic.
The selection rule from the proposal uses the **two-sided** upper endpoint <=16 uM.
No adjustment of alpha, threshold, or q based on candidate count or test coverage.

These are **exploratory conformal-style intervals**, not validated bounds for selected
de novo peptides. Standard split-conformal marginal coverage requires exchangeability;
group splitting, historical mixed assays, generative distribution shift and selection
do not establish that assumption. In particular, 90% marginal coverage does not imply
90% success probability for each selected peptide or >=22/25 successes with 90%
probability. Even independent Bernoulli(0.9) hits do not give the latter statement.
Do not label this implementation conformal risk control for a portfolio.

## Completion of Phase 1

`gflownet.py` upgrades the prior GRU policy with a frozen, stationary multiobjective
reward: log-density from the original small LM, bounded predicted MIC utility,
novelty relative to all reference sequences, repulsion from 512 fixed training anchors,
empirical property agreement, and an exact-known penalty. The density term remains
an explicit regularizer. Trajectory Balance on the append-only length-conditioned tree
has backward probability one. One thousand optimizer steps do not prove convergence
to the target reward distribution. Neither fixed-anchor repulsion nor high batch
uniqueness proves superior diversity. The oracle may be exploited out of distribution.

Diffusion DDIM100 and ProtGPT2-LoRA pools are reused. `finish_library.py` replaces the
GFlowNet branch and writes a separate 50,000-sequence `phase1/artifacts/final/` library,
excluding exact reference, local LM fine-tuning data, and MIC dataset sequences.
All earlier pools and libraries remain intact. Architecture substitutions relative to
the proposal: ESM2-8M rather than a 650M encoder; GRU rather than Transformer GFlowNet;
ProtGPT2-LoRA rather than published AMP-GPT; reference-distance/anchor proxies.

## Ranking and reproducibility

`score.py` caches embeddings and max reference similarity in atomic chunks. For the
competition's top-list novelty rule, it uses normalized **Indel** similarity, equivalent
to the organizer's `python-Levenshtein.ratio`, which differs from normalized edit
similarity used in group splits/reward. Float64 avoids a rounded 0.8 cutoff.
All 50,000 candidates receive predicted historical MIC and exploratory intervals.
If fewer than 100 meet both cutoff rules, no threshold-qualified top100 is emitted;
the diagnostic best100 CSV is explicitly labeled as not threshold-qualified.
Even if 100 pass, their filename says `exploratory`, not certified.

```powershell
phase1/.venv/Scripts/python.exe phase2/download.py
phase1/.venv/Scripts/python.exe phase2/prepare.py
phase1/.venv/Scripts/python.exe phase2/fit.py
phase1/.venv/Scripts/python.exe phase2/gflownet.py train
# Interrupted training: same command plus --resume.
phase1/.venv/Scripts/python.exe phase2/gflownet.py generate
# Interrupted sampling: same command plus --resume.
phase1/.venv/Scripts/python.exe phase2/finish_library.py
phase1/.venv/Scripts/python.exe phase2/score.py
phase1/.venv/Scripts/python.exe phase2/verify.py
phase1/.venv/Scripts/python.exe phase2/report.py
```

The environment is locked; no external data are sent to services. `verify.py` expects
a fresh replay path and tests actual prefix/resume generation, model reload, split
isolation and quantiles. A full cross-hardware library replay and submission entry
point packaging are not established by these prefix checks. Phase 3 clustering,
submodular selection and molecular simulation remain outside this phase.

## Sources

- [GRAMPA authors' repository](https://github.com/zswitten/Antimicrobial-Peptides).
- [Conformal tutorial, Angelopoulos & Bates](https://arxiv.org/abs/2107.07511).
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814): a different guarantee
  requiring its own loss construction and assumptions, not obtained by renaming intervals.
- Organizer novelty implementation: `competition-template/scripts/verify_submission.py`.

Raw data are public research data, not automatically under an OSI license. No root
LICENSE exists in the pinned GRAMPA repository; its constituent database terms remain
unverified for redistribution. Do not claim publication/submission licensing is cleared.
