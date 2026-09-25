from core import *
import argparse
import json
import time


def dataset(tok, name):
    rows = json.loads((OUT / f"data/{name}.json").read_text())
    return [(encode(tok, row["sequence"], row["label"]), row["label"]) for row in rows]


@torch.no_grad()
def evaluate(model, data, tok, bs):
    model.eval()
    scores = {}
    for label in ["AMP", "NONAMP"]:
        examples = [x for x, tag in data if tag == label]
        total, tokens = 0.0, 0
        for i in range(0, len(examples), bs):
            x = batch(examples[i:i+bs], tok)
            count = (x["labels"][:, 1:] != -100).sum().item()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**x).loss
            total += loss.item() * count
            tokens += count
        scores[label + "_token_nll"] = total / tokens
    return scores


def main(args):
    seed(42)
    tok = tokenizer()
    data, val = dataset(tok, "train"), dataset(tok, "validation")
    model = load_model(trainable=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    if args.smoke:
        examples = sorted([x for x, _ in data], key=lambda x: len(x[0]), reverse=True)[:args.batch_size]
        torch.cuda.reset_peak_memory_stats()
        model.train()
        started = time.time()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch(examples, tok)).loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step()
        torch.cuda.synchronize()
        report = {"status": "passed", "model": "nferruz/ProtGPT2 + LoRA r16", "batch_size": args.batch_size, "max_tokens": max(len(x[0]) for x in examples), "trainable_parameters": sum(p.numel() for p in trainable), "total_parameters": sum(p.numel() for p in model.parameters()), "loss": loss.item(), "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20, "seconds": time.time()-started, "gpu": torch.cuda.get_device_name(0)}
        dump(OUT / "smoke.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    directory = OUT / "training"
    directory.mkdir(parents=True, exist_ok=True)
    run_config = {"epochs": args.epochs, "batch_size": args.batch_size, "accumulation": args.accumulation, "lr": args.lr, "seed": 42, "rank": 16, "data_sha256": sha(OUT / "data/train.json"), "base_manifest_sha256": sha(OUT / "base_manifest.json")}
    start_epoch, start_offset, step, best, history = 0, 0, 0, float("inf"), []
    if args.resume:
        state = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
        if state["config"] != run_config:
            raise RuntimeError("Resume config/data mismatch")
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, start_offset, step, best, history = state["epoch"], state["offset"], state["step"], state["best"], state["history"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        if (directory / "resume.pt").exists():
            raise FileExistsError("Training state exists; use --resume")
        baseline = evaluate(model, val, tok, args.batch_size)
        dump(directory / "base_validation.json", baseline)
        print(json.dumps({"stage": "base_validation", **baseline}), flush=True)
    dump(directory / "config.json", run_config)
    from peft import get_peft_model_state_dict
    def checkpoint(epoch, offset):
        temporary = directory / "resume.tmp"
        torch.save({"config": run_config, "adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(), "epoch": epoch, "offset": offset, "step": step, "best": best, "history": history, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(directory / "resume.pt")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    group_size = args.batch_size * args.accumulation
    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng(42 + epoch).permutation(len(data))
        model.train()
        offset = start_offset if epoch == start_epoch else 0
        while offset < len(order):
            group = order[offset:offset + group_size]
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for j in range(0, len(group), args.batch_size):
                selected = group[j:j+args.batch_size]
                x = batch([data[i][0] for i in selected], tok)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**x).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                weight = len(selected) / len(group)
                (loss * weight).backward()
                weighted_loss += loss.item() * weight
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            offset += len(group)
            step += 1
            if step % 10 == 0:
                record = {"epoch": epoch + 1, "step": step, "examples": offset, "total": len(data), "loss": weighted_loss, "elapsed_seconds": time.time()-started, "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
                dump(directory / "progress.json", record)
                print(json.dumps(record), flush=True)
            if step % 20 == 0:
                checkpoint(epoch, offset)
        scores = evaluate(model, val, tok, args.batch_size)
        history.append({"epoch": epoch + 1, **scores})
        print(json.dumps(history[-1]), flush=True)
        if scores["AMP_token_nll"] < best:
            best = scores["AMP_token_nll"]
            model.save_pretrained(OUT / "adapter", safe_serialization=True)
            tok.save_pretrained(OUT / "adapter")
        dump(directory / "history.json", history)
        checkpoint(epoch + 1, 0)
    dump(directory / "complete.json", {"status": "completed", "best_AMP_token_nll": best, "history": history, "trainable_parameters": sum(p.numel() for p in trainable), "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20, "adapter_sha256": sha(OUT / "adapter/adapter_model.safetensors")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    args = p.parse_args()
    if min(args.batch_size, args.accumulation, args.epochs, args.lr) <= 0:
        p.error("All training counts/rates must be positive")
    main(args)
