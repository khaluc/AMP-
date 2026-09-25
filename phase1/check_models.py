"""Check trained-model reproducibility and diagnostics; never interpret as biological validation."""
import json
import numpy as np
import torch
from common import ROOT, WORK, fasta, dump, valid
from networks import PeptideLM, FlowPolicy, seed_all, sample, log_probability, descriptors, trajectory_balance

seed_all(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoints = WORK / "artifacts/checkpoints"
report = {"device": device, "torch": torch.__version__, "checks": {}}
for name, cls in [("lm", PeptideLM), ("gflownet", FlowPolicy)]:
    model = cls().to(device)
    model.load_state_dict(torch.load(checkpoints / f"{name}.pt", map_location=device, weights_only=True))
    seed_all(111)
    first = sample(model, [8, 12, 20, 25, 40, 50] * 8, device)
    seed_all(111)
    second = sample(model, [8, 12, 20, 25, 40, 50] * 8, device)
    assert first == second, f"{name} is not reproducible on the current device"
    assert all(valid(s) for s in first)
    report["checks"][name] = {"same_seed_identical": True, "sample_count": len(first), "unique": len(set(first)), "canonical_and_length_valid": True}
    if name == "lm":
        teacher = model.eval().requires_grad_(False)
    else:
        policy = model.eval()

metadata = json.loads((checkpoints / "gflownet.metadata.json").read_text())
means, scales = np.array(metadata["property_mean"]), np.array(metadata["property_std"])
references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
length_pool = np.array([len(s) for s in fasta(WORK / "artifacts/data/amp_train.fasta")])
rng = np.random.default_rng(999)
lengths = rng.choice(length_pool, 256)
seed_all(999)
sequences = sample(policy, lengths, device)
with torch.no_grad():
    prior, length_tensor = log_probability(teacher, sequences, device)
    z = (descriptors(sequences) - means) / scales
    penalty = torch.tensor(np.minimum(z*z, 9).mean(1), device=device)
    known = torch.tensor([8.0 if s in references else 0.0 for s in sequences], device=device)
    reward = prior - penalty - known
    pf, _ = log_probability(policy, sequences, device)
    mse = trajectory_balance(policy.log_z[length_tensor], pf, reward).item()
report["gflownet_fresh_sample_tb_mse"] = mse
report["interpretation"] = "Finite loss and reproducible valid sampling; not a proof of GFlowNet convergence or AMP activity"
dump(WORK / "artifacts/model_checks.json", report)
print(json.dumps(report, indent=2), flush=True)
