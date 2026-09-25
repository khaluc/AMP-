# Methods and abstract

## Abstract

We constructed an exploratory library of 50,000 unique canonical peptides using three heterogeneous generator families:
published AMP-Diffusion weights sampled with DDIM100, a ProtGPT2 model adapted with LoRA, and a GRU GFlowNet
with a frozen multiobjective surrogate reward. The library combines 16,667, 16,667 and 16,666 accepted sequences,
respectively, with lengths of 8–50 residues and exact exclusion of the antibacterial reference, local language-model
fine-tuning data and the MIC dataset. A random-forest predictor uses ESM2-8M embeddings and ten sequence descriptors
to estimate historical aggregated E. coli log10 MIC. A held-out calibration partition determines exploratory interval
widths without test-based adjustment. Empirical two-sided coverage on 358 test sequences is 77.1% at nominal 90%,
so a 90% coverage or selected-candidate success guarantee is not supported. Among 167 threshold-eligible candidates,
a fixed facility-location plus quality objective selects a 100-member portfolio. Clusters denote geometry rather than
mechanism. The accompanying audit independently reconstructs outputs, checks provenance and predicts with the frozen
model; local seqme diagnostics and the upstream sequence-validation functions provide additional computational checks.
No molecular mechanism, experimentally measured potency, safety or clinical efficacy is established.

## Frozen protocol

- Generator seeds: diffusion 42, ProtGPT2-LoRA 3042, multiobjective GFlowNet 8042; per-round seed increments as recorded.
- Pool batch sizes and sampler settings are taken from each original progress manifest, not changed for performance.
- Predictor selection used validation MAE; final fit used train plus validation. Calibration and test are excluded from fit.
- MIC endpoint: median log10(MIC in micromolar) across available E. coli assays per sequence from the historical dataset.
- Calibration quantile: order statistic ceil((n+1)*0.9) of absolute residuals; separate signed-residual upper diagnostic.
- Eligibility: exploratory two-sided upper endpoint <=16 micromolar and maximum reference Indel similarity <=0.8.
- Portfolio objective: 0.8 times mean centroid coverage plus 0.2 times normalized quality sum / 100; deterministic greedy ties.
- Existing generator-data overlap, assay heterogeneity, group dependence and generative selection shift remain disclosed.

## Interpretation of validation

Recomputing predictions from a verified feature cache is not a fresh ESM embedding run. Checkpoint replay is inference,
not retraining. A repeated same-machine run is not a cross-hardware portability test. Exact novelty is not proof of novel
activity or mechanism. The full original source methods, substitutions and limitations remain in the phase READMEs.
