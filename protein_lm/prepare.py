from core import *
import json

seed(42)
pos_train = fasta(ROOT / "phase1/artifacts/data/amp_train.fasta")
pos_val = fasta(ROOT / "phase1/artifacts/data/amp_val.fasta")
reserved = set(fasta(ROOT / "phase1/artifacts/data/reserved_mic.fasta"))
excluded = set(fasta(ROOT / "data/reference/antibacterial.fasta")) | set(pos_train) | set(pos_val) | reserved
negative_source = ROOT / "starter-kits/hydramp/data/training/Uniprot_0_25_train.csv"
negative = sorted({s for s in csv_sequences(negative_source) if valid(s) and s not in excluded})
neg_train, neg_val = split(negative)
rng = random.Random(42)
rng.shuffle(neg_train)
rng.shuffle(neg_val)
if len(neg_train) < len(pos_train) or len(neg_val) < len(pos_val):
    raise RuntimeError("Not enough disjoint negatives for the requested 1:1 data mix")
neg_train, neg_val = neg_train[:len(pos_train)], neg_val[:len(pos_val)]
tok = tokenizer()
for partition, pos, neg in [("train", pos_train, neg_train), ("validation", pos_val, neg_val)]:
    rows = [{"sequence": s, "label": label} for label, sequences in [("AMP", pos), ("NONAMP", neg)] for s in sequences]
    dump(OUT / f"data/{partition}.json", rows)
assert not (set(pos_train) | set(neg_train)) & (set(pos_val) | set(neg_val))
assert not (set(pos_train) | set(pos_val) | set(neg_train) | set(neg_val)) & reserved
report = {
    "base": "nferruz/ProtGPT2", "positive_train": len(pos_train), "negative_train": len(neg_train),
    "positive_validation": len(pos_val), "negative_validation": len(neg_val),
    "control_prefixes": {label: prefix(tok, label) for label in ["AMP", "NONAMP"]},
    "max_encoded_tokens": max(len(encode(tok, s, label)[0]) for label, seqs in [("AMP", pos_train + pos_val), ("NONAMP", neg_train + neg_val)] for s in seqs),
    "negative_source": str(negative_source.relative_to(ROOT)), "negative_source_sha256": sha(negative_source),
    "positive_source_report_sha256": sha(ROOT / "phase1/artifacts/data/report.json"),
    "limitations": ["Negatives are assumed non-AMP UniProt sequences, not measured inactive controls", "Negatives are mostly <=25 residues, causing possible length confounding", "Exact sequence split, not homology/lab independent", "Unknown overlap with historical ProtGPT2 UniRef50 pretraining; no calibration guarantee"],
}
dump(OUT / "data/report.json", report)
print(json.dumps(report, indent=2))
