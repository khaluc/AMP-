"""Audit exact overlap and split bookkeeping before subsequent calibration work."""
from common import ROOT, WORK, fasta, dump
import json

directory = WORK / "artifacts/data"
train, val, pretrain, preval, reserved = [set(fasta(directory / f"{name}.fasta")) for name in ["amp_train", "amp_val", "uniprot_train", "uniprot_val", "reserved_mic"]]
assert not train & val
assert not pretrain & preval
assert not (train | val | pretrain | preval) & reserved
assert not (pretrain | preval) & val
original_diffusion = set(fasta(ROOT / "starter-kits/ampdiffusion/data/training/training.fasta"))
report = {
    "new_model_exact_split_checks_passed": True,
    "counts": {"amp_train": len(train), "amp_validation": len(val), "uniprot_train": len(pretrain), "uniprot_validation": len(preval), "reserved_mic": len(reserved)},
    "published_diffusion_train_overlap_with_reserved_mic": len(original_diffusion & reserved),
    "uniprot_pretrain_overlap_with_amp_train": len(pretrain & train),
    "limitations": ["Exact overlap only; no homologous-family separation", "Historical checkpoint overlap prevents interpreting all reserved records as unseen by the ensemble", "Assay metadata, censoring, calibration/test separation and selection effects remain Phase 2 work"],
}
dump(directory / "independent_audit.json", report)
print(json.dumps(report, indent=2))
