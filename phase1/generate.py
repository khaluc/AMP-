from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from common import ROOT, WORK, fasta, valid, accept, write_fasta, dump, sha
os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from networks import PeptideLM, FlowPolicy, sample, seed_all


def generate(args):
    seed_all(args.seed)
    output = Path(args.output) if args.output else WORK / "artifacts/pools" / f"{args.model}.fasta"
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".progress.json")
    references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
    seen, collected = set(), []
    start_round = 0
    attempts = 0
    checkpoint = ROOT / "starter-kits/ampdiffusion/checkpoint/model.pt" if args.model == "diffusion" else WORK / f"artifacts/checkpoints/{args.model}.pt"
    settings = {"model": args.model, "seed": args.seed, "batch_size": args.batch_size, "device": args.device, "checkpoint_sha256": sha(checkpoint), "reference_sha256": sha(ROOT / "data/reference/antibacterial.fasta"), "sampler": "ancestral_1000_steps" if args.model == "diffusion" else "categorical_temperature_1"}
    settings.update({"code_sha256": sha(__file__), "networks_sha256": sha(WORK / "networks.py"), "data_report_sha256": sha(WORK / "artifacts/data/report.json"), "torch_version": torch.__version__})
    if args.model == "diffusion" and args.sampler == "ddim":
        settings["sampler"] = f"ddim_eta0_{args.diffusion_steps}_steps"
        settings["ddim_code_sha256"] = sha(WORK / "ddim.py")
    if args.resume and output.exists():
        previous = json.loads(progress.read_text(encoding="utf-8"))
        if previous["settings"] != settings or previous["fasta_sha256"] != sha(output):
            raise ValueError("Resume settings/file mismatch; use a new output path")
        collected = fasta(output)
        if len(collected) != len(set(collected)) or any(not valid(s) or s in references for s in collected):
            raise ValueError("Invalid existing pool")
        seen = set(collected)
        start_round, attempts = previous["next_round"], previous["attempts"]
    elif output.exists():
        raise FileExistsError(f"Use --resume or a new --output: {output}")
    device = torch.device(args.device)
    if args.model == "diffusion":
        sys.path.insert(0, str(ROOT / "starter-kits/ampdiffusion/src"))
        from ampdiffusion_starter_kit.generate import load_model, _decode
        diffusion, esm2, _, indices = load_model(checkpoint, device)
    else:
        model = (PeptideLM() if args.model == "lm" else FlowPolicy()).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        length_pool = np.array([len(s) for s in fasta(WORK / "artifacts/data/amp_train.fasta")])
    started = time.time()
    for round_id in range(start_round, args.max_rounds):
        if len(collected) >= args.count:
            break
        seed_all(args.seed + round_id)
        rng = np.random.default_rng(args.seed + round_id)
        with torch.no_grad():
            if args.model == "diffusion":
                length = int(rng.integers(10, 41))
                if args.sampler == "ddim":
                    from ddim import ddim_sample
                    embeddings = ddim_sample(diffusion, args.batch_size, length + 2, args.diffusion_steps)
                else:
                    embeddings = diffusion.sample(batch_size=args.batch_size, design_len=length + 2)
                sequences = _decode(esm2, embeddings, indices, length)
            else:
                sequences = sample(model, rng.choice(length_pool, args.batch_size), device)
        # Preserve an entire batch so extending a pool does not discard its tail.
        collected.extend(accept(sequences, references, seen, len(sequences)))
        attempts += len(sequences)
        temporary = output.with_suffix(".tmp")
        write_fasta(temporary, collected, args.model)
        temporary.replace(output)
        report = {"settings": settings, "accepted": len(collected), "requested_minimum": args.count, "attempts": attempts, "next_round": round_id + 1, "fasta_sha256": sha(output), "elapsed_this_run_seconds": time.time() - started}
        dump(progress, report)
        print(json.dumps({k: report[k] for k in ["accepted", "requested_minimum", "attempts", "next_round", "elapsed_this_run_seconds"]}), flush=True)
    if len(collected) < args.count:
        raise RuntimeError(f"Only {len(collected)}/{args.count}; max rounds exhausted. No fallback generator used.")


def merge(args):
    names = ["diffusion", "lm", "gflownet"]
    quotas = dict(zip(names, [16667, 16667, 16666]))
    references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
    seen, rows = set(), []
    report = {}
    for name in names:
        path = Path(args.diffusion_pool) if name == "diffusion" and args is not None and args.diffusion_pool else WORK / "artifacts/pools" / f"{name}.fasta"
        sequences = fasta(path)
        kept = accept(sequences, references, seen, quotas[name])
        report[name] = {"pool": len(sequences), "selected": len(kept), "quota": quotas[name], "pool_sha256": sha(path), "pool_path": path.resolve().as_posix()}
        if len(kept) != quotas[name]:
            raise RuntimeError(f"Insufficient unique {name} pool after global deduplication: {len(kept)}/{quotas[name]}. Extend that pool; do not fill with another model.")
        rows.extend((name, s) for s in kept)
    out = WORK / "artifacts/library"
    out.mkdir(parents=True, exist_ok=True)
    write_fasta(out / "library.fasta", [s for _, s in rows])
    with (out / "provenance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "generator", "sequence"])
        for i, (name, sequence) in enumerate(rows, 1):
            writer.writerow([f"seq_{i:06d}", name, sequence])
    report.update({"count": len(rows), "unique": len(seen), "length_histogram": dict(sorted(Counter(len(s) for _, s in rows).items())), "exact_reference_overlap": len(seen & references), "sha256": sha(out / "library.fasta"), "scope": "Phase 1 structural checks only; no potency, top-100, MoA or conformal guarantee"})
    dump(out / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool")
    p.add_argument("model", choices=["diffusion", "lm", "gflownet"])
    p.add_argument("--count", type=int, default=17500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-rounds", type=int, default=2000)
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sampler", choices=["ancestral", "ddim"], default="ancestral")
    p.add_argument("--diffusion-steps", type=int, default=100)
    m = sub.add_parser("merge")
    m.add_argument("--diffusion-pool")
    args = parser.parse_args()
    if args.command == "pool":
        if args.count <= 0 or args.batch_size <= 0 or args.max_rounds <= 0:
            parser.error("count, batch-size and max-rounds must be positive")
        generate(args)
    else:
        merge(args)
