"""Run checkpoint inference, then verify deterministic reconstruction of the frozen release.

The cached mode is explicitly an artifact reconstruction, never a model replay.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .common import dump, read_json, require, sha


def run(command, cwd, log, environment):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {Path(command[2]).name if len(command) > 2 else command[0]}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([str(x) for x in command], cwd=cwd, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({process.returncode}); {log}\n{tail}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generate"))
    parser.add_argument("--mode", choices=["checkpoints", "cached"], default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    require(args.seed == 42, "This release replays frozen seeds 42/3042/8042; other seeds require new scoring")
    if args.root is None:
        candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
        args.root = next((p for p in candidates if (p / "phase1/pyproject.toml").is_file()
                          and (p / "phase2/core.py").is_file()), None)
        require(args.root is not None, "Cannot locate AMP project; supply --root")
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        marker = output / "run.json"
        require(marker.is_file(), "Choose a fresh output directory; previous runs are preserved")
        previous = read_json(marker)
        require(previous.get("status") == "computational_reconstruction_verified"
                and previous.get("mode") == args.mode and previous.get("seed") == args.seed,
                "Choose a fresh output directory for incomplete or different runs")
        for name, key in [("library.fasta", "library_sha256"), ("top.fasta", "top_sha256")]:
            require(sha(output / name) == previous.get(key), "Existing output changed; preserved for inspection")
        repeated = output.with_name(output.name + "-replay-" + uuid.uuid4().hex[:12])
        main(["--root", str(root), "--output", str(repeated), "--mode", args.mode, "--seed", str(args.seed)])
        for name in ["library.fasta", "top.fasta"]:
            require(sha(output / name) == sha(repeated / name), f"Repeated run differs: {name}")
        dump(output / "repeat_check.json", {"byte_identical": True, "repeat_directory": str(repeated),
             "mode": args.mode, "cross_hardware": False})
        print(f"Repeated invocation produced byte-identical outputs: {repeated}", flush=True)
        return
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    p1 = root / "phase1/.venv" / executable
    lm = root / "protein_lm/.venv" / executable
    for project, interpreter in [("phase1", p1), ("protein_lm", lm)]:
        if not interpreter.is_file():
            print(f"Preparing locked {project} environment using uv sync", flush=True)
            subprocess.run(["uv", "sync", "--project", str(root / project), "--locked"], check=True)
        require(interpreter.is_file(), f"Missing interpreter after setup: {interpreter}")
    output.mkdir(parents=True)
    package_src = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONHASHSEED="42")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_src, environment.get("PYTHONPATH")]))
    dump(output / "run.json", {"status": "running", "mode": args.mode, "seed": args.seed})
    try:
        if args.mode == "checkpoints":
            diffusion = read_json(root / "phase1/artifacts/pools/diffusion_ddim100.progress.json")
            settings = diffusion["settings"]
            require(settings["code_sha256"] == sha(root / "phase1/generate.py"), "Diffusion code fingerprint changed")
            ds = output / "pools/diffusion.fasta"
            run([p1, "-u", root / "phase1/generate.py", "pool", "diffusion", "--count", diffusion["accepted"],
                 "--batch-size", settings["batch_size"], "--seed", settings["seed"], "--device", settings["device"],
                 "--sampler", "ddim", "--diffusion-steps", "100", "--output", ds], root,
                output / "logs/diffusion.log", environment)
            language = read_json(root / "protein_lm/artifacts/pool/protgpt2_lora.progress.json")
            ls = language["settings"]
            require(ls["code_sha256"] == sha(root / "protein_lm/sample.py"), "LM code fingerprint changed")
            run([lm, "-u", root / "protein_lm/sample.py", "--count", language["accepted"],
                 "--batch-size", ls["batch_size"], "--seed", ls["seed"], "--top-k", ls["top_k"],
                 "--top-p", ls["top_p"], "--temperature", ls["temperature"],
                 "--output", output / "pools/lm.fasta"], root, output / "logs/lm.log", environment)
            flow = read_json(root / "phase1/artifacts/multiobjective/pool.json")
            require(flow["config"]["code_sha256"] == sha(root / "phase2/gflownet.py"), "GFlowNet code fingerprint changed")
            run([p1, "-u", root / "phase2/gflownet.py", "generate", "--count", flow["accepted"],
                 "--output", output / "pools/gflownet.fasta"], root, output / "logs/gflownet.log", environment)
        command = [p1, "-u", "-m", "amp_completion.audit", "--root", root, "--output", output]
        if args.mode == "checkpoints":
            command += ["--replayed-pools", output / "pools"]
        run(command, root, output / "logs/audit.log", environment)
        dump(output / "run.json", {"status": "computational_reconstruction_verified", "mode": args.mode,
             "seed": args.seed, "library_sha256": sha(output / "library.fasta"),
             "top_sha256": sha(output / "top.fasta"), "model_replay": args.mode == "checkpoints",
             "cross_hardware_replay": False, "scientific_targets_all_met": False})
        print(f"Verified outputs: {output}", flush=True)
    except Exception as error:
        dump(output / "run.json", {"status": "failed", "mode": args.mode, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
