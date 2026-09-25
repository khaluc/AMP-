"""Build a separate 50k library replacing only the small-LM branch, keeping baseline artifacts."""
import sys
import json
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase1"))
from common import fasta, valid, write_fasta, dump, sha

out = ROOT / "protein_lm/artifacts"
reference = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
known_training = {r["sequence"] for partition in ["train", "validation"] for r in json.loads((out / f"data/{partition}.json").read_text())}
sources = [
    ("diffusion_ddim100", ROOT / "phase1/artifacts/pools/diffusion_ddim100.fasta", 16667),
    ("protgpt2_lora", out / "pool/protgpt2_lora.fasta", 16667),
    ("gflownet_baseline", ROOT / "phase1/artifacts/pools/gflownet.fasta", 16666),
]
seen, rows, report = set(), [], {}
for name, path, quota in sources:
    selected = []
    sequences = fasta(path)
    for s in sequences:
        if valid(s) and s not in reference and s not in seen and s not in known_training:
            selected.append(s)
            seen.add(s)
            if len(selected) == quota:
                break
    if len(selected) != quota:
        raise RuntimeError(f"Insufficient {name} pool after all exclusions: {len(selected)}/{quota}")
    rows.extend((name, s) for s in selected)
    report[name] = {"selected": len(selected), "pool_count": len(sequences), "pool_sha256": sha(path)}
directory = out / "library_v2"
write_fasta(directory / "library.fasta", [s for _, s in rows])
with (directory / "provenance.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["id", "generator", "sequence"])
    for i, (name, s) in enumerate(rows, 1):
        writer.writerow([f"seq_{i:06d}", name, s])
sequences = fasta(directory / "library.fasta")
assert len(sequences) == len(set(sequences)) == 50000
assert not set(sequences) & (reference | known_training)
assert all(valid(s) for s in sequences)
report.update({"count": len(sequences), "unique": len(set(sequences)), "exact_reference_overlap": 0, "exact_finetune_data_overlap": 0, "library_sha256": sha(directory / "library.fasta"), "limitations": "Only generator 2 upgraded. GFlowNet still uses the original small-LM proxy reward. No MIC/conformal/top-100 evaluation."})
dump(directory / "audit.json", report)
print(json.dumps(report, indent=2))
