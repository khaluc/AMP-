"""Autoregressive BPE sampling with amino-acid length limits, provenance and resumable rounds."""
from core import *
import argparse
import json
import time


def token_table(tok):
    values = []
    for i in range(len(tok)):
        raw = tok.decode([i], clean_up_tokenization_spaces=False)
        seq = "".join(raw.split())
        values.append(seq if seq and set(seq) <= set(AA) else "")
    return values


@torch.no_grad()
def draw(model, tok, texts, bs=32, top_k=100, top_p=0.95, temperature=1.0):
    model.eval()
    p = prefix(tok, "AMP")
    ids = torch.tensor([p] * bs, device="cuda")
    lengths = torch.zeros(bs, dtype=torch.long, device="cuda")
    token_lengths = torch.tensor([len(t) for t in texts], device="cuda")
    eos = tok.eos_token_id
    finished = torch.zeros(bs, dtype=torch.bool, device="cuda")
    attention = torch.ones_like(ids)
    past = None
    emitted = []
    for _ in range(51):
        output = model(input_ids=ids if past is None else ids[:, -1:], attention_mask=attention, past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1].float() / temperature
        allowed = (token_lengths[None] > 0) & (lengths[:, None] + token_lengths[None] <= 50)
        allowed[:, eos] = lengths >= 8
        allowed[finished] = False
        allowed[finished, eos] = True
        logits.masked_fill_(~allowed, -float("inf"))
        if top_k:
            cutoff = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
            logits.masked_fill_(logits < cutoff, -float("inf"))
        sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
        # PyTorch 2.5 CUDA float cumsum has no deterministic implementation.
        # Keep strict determinism: only this cumulative sum runs on CPU.
        cumulative = sorted_logits.softmax(-1).cpu().cumsum(-1).to(logits.device)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits.masked_fill_(torch.zeros_like(remove).scatter(1, sorted_idx, remove), -float("inf"))
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("No finite generation distribution")
        nxt = torch.multinomial(probabilities, 1).squeeze(1)
        emitted.append(nxt.cpu().tolist())
        lengths += torch.where((nxt == eos) | finished, 0, token_lengths[nxt])
        finished |= nxt == eos
        ids = torch.cat([ids, nxt[:, None]], 1)
        attention = torch.cat([attention, torch.ones(bs, 1, device="cuda", dtype=torch.long)], 1)
        if finished.all():
            break
    sequences = []
    for row in zip(*emitted):
        parts = []
        for token in row:
            if token == eos:
                break
            parts.append(texts[token])
        sequences.append("".join(parts))
    if not all(valid(s) for s in sequences):
        raise RuntimeError("Sampler emitted invalid sequence")
    return sequences


def main(args):
    seed(args.seed)
    tok = tokenizer()
    model = load_model(OUT / "adapter")
    texts = token_table(tok)
    references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
    # Also reject known training/validation peptides, including those absent from the official reference.
    for partition in ["train", "validation"]:
        references.update(r["sequence"] for r in json.loads((OUT / f"data/{partition}.json").read_text()))
    settings = {"seed": args.seed, "batch_size": args.batch_size, "top_k": args.top_k, "top_p": args.top_p, "temperature": args.temperature, "adapter_sha256": sha(OUT / "adapter/adapter_model.safetensors"), "base_manifest_sha256": sha(OUT / "base_manifest.json"), "code_sha256": sha(__file__), "core_sha256": sha(HERE / "core.py"), "reference_sha256": sha(ROOT / "data/reference/antibacterial.fasta"), "train_sha256": sha(OUT / "data/train.json"), "validation_sha256": sha(OUT / "data/validation.json")}
    path = Path(args.output) if args.output else OUT / "pool/protgpt2_lora.fasta"
    progress = path.with_suffix(".progress.json")
    collected, seen, start_round, attempts = [], set(), 0, 0
    if path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume or choose another path")
        previous = json.loads(progress.read_text())
        if previous["settings"] != settings or previous["sha256"] != sha(path):
            raise RuntimeError("Resume fingerprint mismatch")
        collected = fasta(path)
        seen = set(collected)
        if len(seen) != len(collected) or seen & references or not all(valid(s) for s in seen):
            raise RuntimeError("Invalid saved pool")
        start_round, attempts = previous["next_round"], previous["attempts"]
    started = time.time()
    for round_id in range(start_round, args.max_rounds):
        if len(collected) >= args.count:
            break
        seed(args.seed + round_id)
        sequences = draw(model, tok, texts, args.batch_size, args.top_k, args.top_p, args.temperature)
        attempts += len(sequences)
        for s in sequences:
            if s not in references and s not in seen:
                seen.add(s)
                collected.append(s)
        temporary = path.with_suffix(".tmp")
        write_fasta(temporary, collected, "protgpt2_lora")
        temporary.replace(path)
        report = {"settings": settings, "accepted": len(collected), "requested": args.count, "attempts": attempts, "next_round": round_id + 1, "sha256": sha(path), "elapsed_this_run_seconds": time.time()-started, "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        dump(progress, report)
        if round_id % 10 == 0 or len(collected) >= args.count:
            print(json.dumps({k: report[k] for k in ["accepted", "attempts", "next_round", "elapsed_this_run_seconds", "peak_reserved_mib"]}), flush=True)
    if len(collected) < args.count:
        raise RuntimeError("Pool quota not reached; no fallback generator used")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=17500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=3042)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-rounds", type=int, default=10000)
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.count, args.batch_size, args.max_rounds, args.temperature) <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        p.error("Invalid generation parameters")
    main(args)
