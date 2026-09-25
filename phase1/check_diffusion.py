import os
import sys
import time
import json
from common import ROOT, WORK, dump, fasta, valid, write_fasta
os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from networks import seed_all, descriptors
from ddim import ddim_sample
sys.path.insert(0, str(ROOT / "starter-kits/ampdiffusion/src"))
from ampdiffusion_starter_kit.generate import load_model, _decode

device = torch.device("cuda")
model, esm2, _, indices = load_model(ROOT / "starter-kits/ampdiffusion/checkpoint/model.pt", device)
length = int(np.random.default_rng(42).integers(10, 41))
outputs = []
timings = []
for _ in range(2):
    seed_all(42)
    start = time.time()
    embeddings = ddim_sample(model, 64, length + 2, steps=100)
    sequences = _decode(esm2, embeddings, indices, length)
    timings.append(time.time() - start)
    outputs.append(sequences)
assert outputs[0] == outputs[1]
assert all(valid(s) for s in outputs[0])
baseline = fasta(WORK / "artifacts/pools/diffusion.fasta")
references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
write_fasta(WORK / "artifacts/experiments/ddim_smoke64.fasta", outputs[0], "ddim_smoke")
report = {
    "variant": "DDIM eta=0, 100 inference steps, original 1000-training-timestep checkpoint",
    "seed": 42, "batch_size": 64, "length": length, "same_seed_identical": True,
    "valid": all(valid(s) for s in outputs[0]), "unique": len(set(outputs[0])),
    "exact_reference_overlap": len(set(outputs[0]) & references),
    "seconds_per_batch": timings,
    "descriptor_order": ["charge_density", "hydrophobic_fraction", "adjacent_repeat_fraction"],
    "ddim_descriptor_means": descriptors(outputs[0]).mean(0).tolist(),
    "ancestral_descriptor_means": descriptors(baseline).mean(0).tolist(),
    "limitations": "Small single-length diagnostic; does not establish equivalent potency, diversity or score to ancestral sampling",
}
dump(WORK / "artifacts/diffusion_checks.json", report)
print(json.dumps(report, indent=2), flush=True)
