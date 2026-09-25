from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import torch
from torch import nn
from common import AA

BOS = 20
PAD = 21


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Avoid hardware-dependent attention backend selection for reproducibility.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def encode(sequences, device):
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    tokens = torch.full((len(sequences), int(lengths.max()) + 1), PAD, dtype=torch.long, device=device)
    tokens[:, 0] = BOS
    for i, sequence in enumerate(sequences):
        tokens[i, 1:len(sequence) + 1] = torch.tensor([AA.index(a) for a in sequence], device=device)
    return tokens[:, :-1], tokens[:, 1:], lengths


class PeptideLM(nn.Module):
    """Small length-conditioned causal Transformer, locally pretrained then AMP-finetuned."""
    def __init__(self, width=128, layers=2):
        super().__init__()
        self.token = nn.Embedding(22, width)
        self.position = nn.Embedding(51, width)
        self.length = nn.Embedding(51, width)
        block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, 20)

    def forward(self, tokens, lengths):
        n = tokens.shape[1]
        hidden = self.token(tokens) + self.position(torch.arange(n, device=tokens.device))[None] + self.length(lengths)[:, None]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), diagonal=1)
        return self.head(self.encoder(hidden, mask=mask))


class FlowPolicy(nn.Module):
    """Append-only sequence tree, fixed length context, deterministic backward policy."""
    def __init__(self, width=128):
        super().__init__()
        self.token = nn.Embedding(22, width)
        self.length = nn.Embedding(51, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, 20)
        self.log_z = nn.Parameter(torch.zeros(51))

    def forward(self, tokens, lengths):
        hidden, _ = self.rnn(self.token(tokens), self.length(lengths).unsqueeze(0))
        return self.head(hidden)


def log_probability(model, sequences, device):
    inputs, targets, lengths = encode(sequences, device)
    mask = targets != PAD
    logp = model(inputs, lengths).log_softmax(-1)
    chosen = logp.gather(-1, targets.clamp(max=19).unsqueeze(-1)).squeeze(-1)
    return (chosen * mask).sum(-1), lengths


@torch.no_grad()
def sample(model, lengths, device, temperature=1.0):
    model.eval()
    lengths = torch.as_tensor(lengths, device=device)
    tokens = torch.full((len(lengths), 1), BOS, device=device, dtype=torch.long)
    hidden = model.length(lengths).unsqueeze(0) if isinstance(model, FlowPolicy) else None
    for _ in range(int(lengths.max())):
        if isinstance(model, FlowPolicy):
            output, hidden = model.rnn(model.token(tokens[:, -1:]), hidden)
            logits = model.head(output[:, -1])
        else:
            logits = model(tokens, lengths)[:, -1]
        nxt = torch.multinomial((logits / temperature).softmax(-1), 1)
        tokens = torch.cat([tokens, nxt], dim=1)
    rows = tokens[:, 1:].cpu().tolist()
    return ["".join(AA[a] for a in row[:int(length)]) for row, length in zip(rows, lengths.cpu().tolist())]


def descriptors(sequences):
    # Simple empirical descriptors, NOT mechanistic labels or MIC estimates.
    result = []
    for s in sequences:
        n = len(s)
        charge = (s.count("K") + s.count("R") - s.count("D") - s.count("E")) / n
        hydrophobic = sum(a in "AILMFWVY" for a in s) / n
        repeats = sum(a == b for a, b in zip(s, s[1:])) / max(1, n - 1)
        result.append([charge, hydrophobic, repeats])
    return np.asarray(result, dtype=np.float32)


def trajectory_balance(log_z, log_pf, log_reward):
    # P_B=1 on an append-only tree. Length is a context, not a learned action.
    return (log_z + log_pf - log_reward).square().mean()
