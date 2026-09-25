"""Check actual adapter sampling and resume against production, then emit an evidence-based report."""
from core import *
from sample import main as generate
from types import SimpleNamespace
import json

source = OUT / "pool/protgpt2_lora.fasta"
settings = json.loads(source.with_suffix(".progress.json").read_text())["settings"]
target = OUT / "replay/prefix.fasta"
args = SimpleNamespace(count=1, batch_size=settings["batch_size"], seed=settings["seed"], top_k=settings["top_k"], top_p=settings["top_p"], temperature=settings["temperature"], max_rounds=10000, output=str(target), resume=False)
generate(args)
first = fasta(target)
assert first == fasta(source)[:len(first)]
args.count = len(first) + 1
args.resume = True
generate(args)
second = fasta(target)
assert second == fasta(source)[:len(second)]
dump(OUT / "replay_checks.json", {"same_seed_production_prefix": True, "resume_prefix": True, "checked_sequences": len(second), "scope": "Same hardware/software, first two accepted rounds; not a full library replay"})
print("Production-prefix replay and resume verified.", flush=True)
