"""Independent reconstruction of library, predictions, intervals and portfolio.

Existing models and thresholds are frozen. Nothing is tuned using the test labels.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Indel
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from .common import assemble, dump, fasta, read_json, require, sequences, sha, valid, write_fasta


def quantile(scores, alpha=0.1):
    scores = np.asarray(scores, dtype=float)
    require(scores.ndim == 1 and len(scores) > 0 and np.isfinite(scores).all(), "Invalid calibration scores")
    require(0 < alpha < 1, "Invalid alpha")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if rank > len(scores) else float(np.sort(scores)[rank - 1])


def fingerprint_manifest(root, manifest):
    checked = {}
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Manifest path escapes project")
        actual = sha(path)
        require(actual == expected, f"Changed source/artifact: {relative}")
        checked[relative] = actual
    return checked


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def audit(root, output, replayed_pools=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    p2, p3 = root / "phase2/artifacts", root / "phase3/artifacts"
    checked = fingerprint_manifest(root, read_json(p2 / "run_manifest.json"))
    p3_manifest = read_json(p3 / "manifest.json")
    for name, expected in p3_manifest["scripts"].items():
        require(sha(root / "phase3" / name) == expected, f"Phase 3 source changed: {name}")
    require(sha(p2 / "ranked_library.csv") == p3_manifest["ranked_sha256"], "Ranked table changed")
    original_library = root / "phase1/artifacts/final/library.fasta"
    require(sha(original_library) == p3_manifest["library_sha256"], "Library changed")
    original_top = p3 / "top100_portfolio_exploratory.fasta"
    require(sha(original_top) == p3_manifest["top100_sha256"], "Portfolio changed")
    reference = sequences(root / "data/reference/antibacterial.fasta")
    data = pd.read_csv(p2 / "dataset.csv")
    known = set(reference) | set(data.sequence)
    for partition in ["train", "validation"]:
        known.update(row["sequence"] for row in read_json(root / f"protein_lm/artifacts/data/{partition}.json"))
    source_paths = [root / "phase1/artifacts/pools/diffusion_ddim100.fasta",
                    root / "protein_lm/artifacts/pool/protgpt2_lora.fasta",
                    root / "phase1/artifacts/multiobjective/pool.fasta"]
    pool_checks = {}
    pools = []
    final_audit = read_json(root / "phase1/artifacts/final/audit.json")
    for name, source, generator in zip(["diffusion", "lm", "gflownet"], source_paths,
            ["diffusion_ddim100", "protgpt2_lora", "gflownet_multiobjective"]):
        require(sha(source) == final_audit["sources"][generator]["pool_sha256"], f"Changed {name} source pool")
        records = sequences(source)
        if replayed_pools:
            replay = Path(replayed_pools) / f"{name}.fasta"
            regenerated = sequences(replay)
            require(regenerated == records, f"Full checkpoint replay mismatch: {name}")
            pool_checks[name] = {"count": len(records), "all_sequences_in_order_match": True,
                                 "sha256": sha(replay)}
            records = regenerated
        pools.append(records)
    library = assemble(pools, known)
    require(len(library) == len(set(library)) == 50000, "Invalid final count/uniqueness")
    records = [(f"seq_{i:06d}", sequence) for i, sequence in enumerate(library, 1)]
    require(records == fasta(original_library), "Reassembled full library differs")
    write_fasta(output / "library.fasta", records)
    require(sha(output / "library.fasta") == sha(original_library), "Library byte replay mismatch")
    print("Library: all 50,000 sequences reconstructed and verified", flush=True)

    ranked = pd.read_csv(p2 / "ranked_library.csv").sort_values("id").reset_index(drop=True)
    require(ranked.sequence.tolist() == library, "Score rows do not align with library")
    cache = p2 / "library_features"
    require(read_json(cache / "manifest.json") == {"library": sha(original_library),
        "core": sha(root / "phase2/core.py"), "model": sha(p2 / "mic_regressor.joblib")}, "Feature-cache provenance differs")
    blocks = []
    for offset in range(0, 50000, 512):
        path = cache / f"{offset:06d}.npz"
        require(sha(path) == p3_manifest["feature_chunks"][path.name], f"Feature cache changed: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            features = saved["features"]
        require(features.shape == (min(512, 50000-offset), 330) and np.isfinite(features).all(), "Invalid feature block")
        blocks.append(features)
    x = np.concatenate(blocks)
    model = joblib.load(p2 / "mic_regressor.joblib")
    with threadpool_limits(limits=4):
        predictions = model.predict(x)
    require(np.allclose(predictions, ranked.prediction_log10_uM, atol=1e-10, rtol=0), "Recomputed predictions differ")
    calibration = read_json(p2 / "calibration.json")
    heldout = pd.read_csv(p2 / "heldout_predictions.csv")
    require(heldout.sequence.tolist() == data.sequence.tolist(), "Heldout rows differ from dataset")
    data_x = np.load(p2 / "dataset_features.npy", allow_pickle=False)
    require(data_x.shape == (len(data), 330) and np.isfinite(data_x).all(), "Invalid dataset features")
    data_pred = model.predict(data_x)
    require(np.allclose(data_pred, heldout.prediction_log10_uM, atol=1e-10, rtol=0), "Heldout prediction replay differs")
    y = data.target_log10_uM.to_numpy()
    cal = data.split.eq("calibration").to_numpy()
    q = quantile(abs(y[cal] - data_pred[cal]))
    qu = quantile(y[cal] - data_pred[cal])
    require(math.isclose(q, calibration["q_two_sided"], abs_tol=1e-12), "Conformal quantile differs")
    require(math.isclose(qu, calibration["q_one_sided"], abs_tol=1e-12), "One-sided quantile differs")
    require(np.allclose(ranked.upper_uM, 10**(predictions+q), rtol=1e-10), "Candidate intervals differ")
    require(np.allclose(ranked.lower_log10_uM, predictions-q, atol=1e-10), "Candidate lower bounds differ")
    diagnostics = {}
    for split in ["train", "validation", "calibration", "test"]:
        mask = data.split.eq(split).to_numpy()
        residual = y[mask] - data_pred[mask]
        diagnostics[split] = {"n": int(mask.sum()), "mae_log10": float(abs(residual).mean()),
            "mean_signed_residual_log10": float(residual.mean()),
            "covered_two_sided": int((abs(residual) <= q).sum()),
            "coverage_two_sided": float((abs(residual) <= q).mean()),
            "coverage_one_sided": float((residual <= qu).mean())}
    # These data have already been inspected; never retune a new method to their test coverage.
    stats = {"nominal_coverage": 0.9, "q_two_sided": q, "q_one_sided": qu,
        "by_split": diagnostics, "statistical_target_met": diagnostics["test"]["coverage_two_sided"] >= 0.9,
        "parameters_retuned": False, "new_independent_test_required_for_next_model": True,
        "refit_development_splits": ["train", "validation"], "validation_residuals_are_in_sample_after_refit": True,
        "scope": "Historical aggregated E. coli MIC; no per-candidate, selected-portfolio, or experimental hit-rate guarantee."}
    dump(output / "calibration_audit.json", stats)
    print("Predictions and calibration: recomputed with the frozen model", flush=True)

    clustering = joblib.load(p3 / "clustering.joblib")
    z = np.concatenate([clustering["embedding_scaler"].transform(x[:, :320]) * math.sqrt(.5/320),
                        clustering["descriptor_scaler"].transform(x[:, 320:]) * math.sqrt(.5/10)], axis=1)
    km = clustering["clustering"]
    mask = (ranked.upper_uM <= 16) & (ranked.max_reference_similarity <= .8)
    eligible = ranked[mask].reset_index(drop=True)
    saved_eligible = pd.read_csv(p3 / "eligible_with_clusters.csv")
    require(eligible.id.tolist() == saved_eligible.id.tolist(), "Eligible set differs")
    result = read_json(p3 / "result.json")
    bandwidth = result["config"]["rbf_bandwidth"]
    with threadpool_limits(limits=4):
        distance = pairwise_distances(km.cluster_centers_, z[mask], metric="sqeuclidean")
    similarity = np.exp(-distance / (2*bandwidth**2))
    quality = eligible.upper_uM.min() / eligible.upper_uM.to_numpy()
    with np.load(p3 / "selection_inputs.npz", allow_pickle=False) as saved:
        require(np.allclose(similarity, saved["similarity"], atol=1e-10), "Recomputed portfolio similarities differ")
        require(np.allclose(quality, saved["quality"], atol=1e-12), "Portfolio quality differs")
    selection = module("amp_frozen_selection", root / "phase3/selection.py")
    chosen, trace = selection.greedy(similarity, quality, 100, .8)
    top = eligible.iloc[chosen].copy()
    require(list(zip(top.id, top.sequence)) == fasta(original_top), "Recomputed portfolio differs")
    maxima = []
    for offset in range(0, len(top), 10):
        values = process.cdist(top.sequence.tolist()[offset:offset+10], reference,
                              scorer=Indel.normalized_similarity, dtype=np.float64, workers=4)
        maxima.extend(values.max(axis=1).tolist())
    require(max(maxima) <= .8, "A top peptide exceeds the reference similarity limit")
    require(np.allclose(maxima, top.max_reference_similarity, atol=1e-12, rtol=0), "Stored novelty differs")
    write_fasta(output / "top.fasta", zip(top.id, top.sequence))
    require(sha(output / "top.fasta") == sha(original_top), "Top-list byte replay differs")
    top.insert(0, "portfolio_rank", range(1, 101))
    top["selection_gain"] = [row["marginal_gain"] for row in trace]
    top["mechanism_status"] = "unassigned_no_mechanistic_validation"
    top.to_csv(output / "top.csv", index=False)
    dump(output / "greedy_trace.json", trace)
    audit_report = {"library_count": len(library), "unique": len(set(library)),
        "top_count": len(top), "top_max_reference_similarity": max(maxima),
        "exact_known_overlap": len(set(library) & known), "library_sha256": sha(output / "library.fasta"),
        "top_sha256": sha(output / "top.fasta"), "recomputed_model_predictions": len(predictions),
        "portfolio_recomputed_from_features": True, "source_fingerprints_checked": checked,
        "checkpoint_pool_replay": pool_checks, "cross_hardware_replay": False,
        "statistical_target_met": stats["statistical_target_met"], "biological_activity_validated": False,
        "official_aggregation_score": None, "mechanistic_simulations_completed": False}
    dump(output / "audit.json", audit_report)
    print("Portfolio: all 100 members, order, bounds and reference similarity verified", flush=True)
    return audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replayed-pools", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output, args.replayed_pools)


if __name__ == "__main__":
    main()
