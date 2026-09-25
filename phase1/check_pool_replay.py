"""Replay production batch prefixes, including a resume boundary, with saved checkpoints."""
from pathlib import Path
from types import SimpleNamespace
import json
from common import WORK, fasta, dump
from generate import generate

report = {}
for name, filename in [("lm", "lm"), ("gflownet", "gflownet"), ("diffusion", "diffusion_ddim100")]:
    source = WORK / f"artifacts/pools/{filename}.fasta"
    settings = json.loads(source.with_suffix(".progress.json").read_text())["settings"]
    target = WORK / f"artifacts/replay/{filename}.fasta"
    if target.exists():
        raise FileExistsError(f"Replay output already exists; preserve it and select a fresh output directory: {target}")
    args = SimpleNamespace(model=name, count=1, batch_size=settings["batch_size"], seed=settings["seed"], device=settings["device"], max_rounds=2000, output=str(target), resume=False, sampler="ddim" if name == "diffusion" else "ancestral", diffusion_steps=100)
    generate(args)
    first = fasta(target)
    assert first == fasta(source)[:len(first)], f"First production batch changed for {name}"
    args.resume = True
    args.count = len(first) + 1
    generate(args)
    resumed = fasta(target)
    assert resumed == fasta(source)[:len(resumed)], f"Resume did not reproduce production prefix for {name}"
    report[name] = {"first_batch_match": True, "resumed_prefix_match": True, "prefix_sequences": len(resumed), "batch_size": args.batch_size, "device": args.device}
    event_path = WORK / "artifacts/resume_event.json"
    if name == "diffusion" and event_path.exists():
        event = json.loads(event_path.read_text())
        offset = event["accepted_before_resume"]
        production = fasta(source)
        args.seed = event["seed"] + event["completed_rounds_before_resume"]
        args.output = str(WORK / "artifacts/replay/diffusion_actual_resume.fasta")
        args.resume = False
        args.count = 1
        generate(args)
        earlier = set(production[:offset])
        resumed_batch = [s for s in fasta(args.output) if s not in earlier]
        assert resumed_batch == production[offset:offset + len(resumed_batch)], "Actual interruption boundary did not replay"
        report[name]["actual_interruption_boundary_match"] = True
        report[name]["actual_interruption_replayed_sequences"] = len(resumed_batch)
    dump(WORK / "artifacts/production_replay_checks.json", report)
print(json.dumps(report, indent=2), flush=True)
