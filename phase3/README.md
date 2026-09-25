# Phase 3: exploratory submodular portfolio selection

This phase selects 100 sequences from the 167 eligible Phase 2 candidates. The
exploratory two-sided upper bound remains <=16 uM and maximum reference similarity
remains <=0.8 using normalized Indel similarity (organizer's Levenshtein.ratio).
The earlier question about 64 uM did not change the selection threshold.

## Representation and clusters

Reuse the checked, ordered ESM2-8M residue-mean embeddings and ten descriptors for
all 50,000 final library sequences. Verify their manifest, shapes, finite values and
descriptor alignment with FASTA before using them. Standardize the 320 embedding
dimensions and 10 descriptor dimensions separately; scale by sqrt(0.5/320) and
sqrt(0.5/10), respectively. Thus neither block dominates merely because it is wider.

Fit MiniBatchKMeans with 100 clusters, seed42, n_init10, batch2048, max_iter200 and
four CPU threads. This uses already computed features, so no new GPU inference is
needed. Clusters describe sequence geometry, **not mechanism of action**. The full
library is used unsupervised; no calibration/test activity labels are used in this phase.

## Selection objective

For each of the 100 centroids c and eligible candidate x, define
`s(c,x)=exp(-||c-x||^2/(2*b^2))`, where b is the median nearest-other-centroid distance.
All centroids have equal weight, to represent small as well as large clusters.
Define nonnegative candidate quality `q(x)=min_candidate_upper_MIC/upper_MIC(x)`.
The weights below were fixed before observing selection results:

`F(S)=0.8 * mean_c max_{x in S} s(c,x) + 0.2/100 * sum_{x in S} q(x)`.

F(empty)=0. The first term is facility-location coverage and the second a modular
quality term. Both are nonnegative and monotone; their sum is submodular. Standard
greedy adds the maximum marginal-gain candidate until |S|=100, with stable sequence
ID order breaking exact ties. The cardinality-constrained greedy approximation
bound applies to **this fixed surrogate objective only**, not to MIC success, diversity
of mechanisms or clinical efficacy. No generator quotas or extra matroid constraints
are imposed. Greedy trace and numeric inputs are exported for independent replay.

The proposal's 'one from every cluster' rule is infeasible if some global clusters
contain no threshold-eligible candidate. We keep the original eligibility threshold
and use soft facility-location coverage rather than silently claiming 100 represented
clusters or relaxing activity/novelty constraints. The report gives both available and
selected hard-cluster counts. Soft kernel coverage is not the percentage of MoAs covered.

## Evaluation and limits

Compare against Phase 2's MIC-ranked top100 using the same eligible set and objective.
Report generator counts, hard clusters, kernel coverage, pairwise normalized Indel
distance, pairs above 80% similarity, and upper-MIC distribution. Simulate 2,000 draws
of 25 without replacement using seed3042 to describe **cluster coverage only**;
no predicted hit/miss outcomes or MoA labels are invented.

All outputs remain exploratory: Phase 2 test coverage was 77.1% at nominal90% and
substantial overlap with published diffusion pretraining remains. Phase 3 cannot repair
those calibration assumptions. Representatives are listed for future research, with
mechanisms explicitly unassigned. Optional MARTINI simulations are **not performed**:
sequence clustering alone cannot establish carpet/pore/intracellular action. No
membrane compositions, validated structures/topologies or mechanistic evidence have
been established here. This deliverable completes computational portfolio selection,
not optional molecular-dynamics validation or experimental validation.

## Run

Use the existing locked environment:

```powershell
uv sync --project phase1 --locked
phase1/.venv/Scripts/python.exe -m pytest phase3/test_selection.py -q -p no:cacheprovider
phase1/.venv/Scripts/python.exe -u phase3/run.py
phase1/.venv/Scripts/python.exe phase3/report.py
```

The old Phase2 top100 remains unchanged. Main new output:
`artifacts/top100_portfolio_exploratory.fasta` / `.csv`, with original library IDs.
FASTA order follows greedy selection order; this is a portfolio construction order,
not a ranking of individual potency. Report: `KET_QUA.md`.

References:
- [Nemhauser and Wolsey, cardinality-constrained submodular maximization](https://pubsonline.informs.org/doi/abs/10.1287/moor.3.3.177).
- [Facility-location and greedy approximation](https://proceedings.mlr.press/v97/elhamifar19a/elhamifar19a.pdf).
