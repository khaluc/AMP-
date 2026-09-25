from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import WORK, ROOT, fasta, dump, sha
from networks import PeptideLM, FlowPolicy, PAD, encode, seed_all, sample, log_probability, descriptors, trajectory_balance


def supervised(model, train, val, epochs, device, label, output, lr=0.0003, batch_size=128):
    print(json.dumps({"stage": label, "status": "starting", "train": len(train), "validation": len(val), "epochs": epochs}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    best, history = float("inf"), []
    for epoch in range(epochs):
        started = time.time()
        model.train()
        totals, count = 0.0, 0
        order = rng.permutation(len(train))
        for offset in range(0, len(order), batch_size):
            sequences = [train[i] for i in order[offset:offset + batch_size]]
            inputs, targets, lengths = encode(sequences, device)
            loss = F.cross_entropy(model(inputs, lengths).reshape(-1, 20), targets.reshape(-1), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += loss.item() * len(sequences)
            count += len(sequences)
            if offset // batch_size % 200 == 0:
                print(json.dumps({"stage": label, "epoch": epoch + 1, "processed": count, "total": len(train), "loss": loss.item()}), flush=True)
        model.eval()
        nll, tokens = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(val), batch_size):
                lp, lengths = log_probability(model, val[offset:offset + batch_size], device)
                nll -= lp.sum().item()
                tokens += lengths.sum().item()
        score = nll / tokens
        row = {"stage": label, "epoch": epoch + 1, "train_batch_mean_nll": totals / count, "validation_token_nll": score, "seconds": time.time() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), output)
        dump(output.with_suffix(".history.json"), history)
    model.load_state_dict(torch.load(output, map_location=device, weights_only=True))
    return history


def train(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    out = WORK / "artifacts/checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    data = WORK / "artifacts/data"
    amp = fasta(data / "amp_train.fasta")
    val = fasta(data / "amp_val.fasta")
    if args.stage == "lm":
        model = PeptideLM().to(device)
        supervised(model, fasta(data / "uniprot_train.fasta"), fasta(data / "uniprot_val.fasta"), args.pretrain_epochs, device, "uniprot_pretrain", out / "lm_pretrained.pt")
        supervised(model, amp, val, args.epochs, device, "amp_finetune", out / "lm.pt", lr=0.0001)
        dump(out / "lm.metadata.json", {"architecture": "128-wide 2-layer length-conditioned causal Transformer", "parameters": sum(p.numel() for p in model.parameters()), "seed": args.seed, "pretraining": "local UniProt peptides; not published AMP-GPT", "data_report_sha256": sha(data / "report.json"), "checkpoint_sha256": sha(out / "lm.pt")})
        return
    teacher = PeptideLM().to(device)
    teacher.load_state_dict(torch.load(out / "lm.pt", map_location=device, weights_only=True))
    teacher.eval().requires_grad_(False)
    policy = FlowPolicy().to(device)
    supervised(policy, amp, val, args.warmup_epochs, device, "gflownet_supervised_warmup", out / "gflownet_warmup.pt")
    means = descriptors(amp).mean(0)
    scales = descriptors(amp).std(0).clip(0.05)
    references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
    lengths_pool = np.array([len(s) for s in amp])
    optimizer = torch.optim.Adam([
        {"params": [p for name, p in policy.named_parameters() if name != "log_z"], "lr": 0.0001},
        {"params": [policy.log_z], "lr": 0.01},
    ])
    rng = np.random.default_rng(args.seed + 1)
    history = []
    started = time.time()
    for step in range(args.steps):
        sequences = sample(policy, rng.choice(lengths_pool, args.batch_size), device)
        with torch.no_grad():
            prior, lengths = log_probability(teacher, sequences, device)
            z = (descriptors(sequences) - means) / scales
            # Multi-objective positive reward: LM density prior, empirical property
            # agreement, and exact-reference novelty. Does NOT estimate potency.
            property_penalty = torch.tensor(np.minimum(z * z, 9).mean(1), device=device)
            known_penalty = torch.tensor([8.0 if s in references else 0.0 for s in sequences], device=device)
            log_reward = prior - property_penalty - known_penalty
        policy.train()
        log_pf, lengths = log_probability(policy, sequences, device)
        loss = trajectory_balance(policy.log_z[lengths], log_pf, log_reward)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite trajectory-balance loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "trajectory_balance_mse": loss.item(), "mean_log_reward": log_reward.mean().item(), "batch_unique_fraction": len(set(sequences)) / len(sequences), "seconds": time.time() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            torch.save(policy.state_dict(), out / "gflownet.pt")
            dump(out / "gflownet.history.json", history)
    dump(out / "gflownet.metadata.json", {"architecture": "128-wide GRU, append-only length-conditioned trajectory balance", "seed": args.seed, "steps": args.steps, "reward": "exp(LM log probability - empirical property penalty - 8*exact_known)", "property_mean": means.tolist(), "property_std": scales.tolist(), "teacher_sha256": sha(out / "lm.pt"), "checkpoint_sha256": sha(out / "gflownet.pt"), "limitations": "Approximate reward sampling; no MIC, MoA, coverage or superiority claim"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lm", "gflownet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.epochs, args.warmup_epochs, args.steps, args.batch_size) <= 0:
        parser.error("epoch, step and batch counts must be positive")
    train(args)
