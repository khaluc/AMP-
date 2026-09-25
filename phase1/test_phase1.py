import numpy as np
import pytest
import torch
from common import AA, accept, split, valid
from networks import PeptideLM, FlowPolicy, seed_all, sample, log_probability, trajectory_balance


def test_reference_and_global_duplicates():
    known = {"ACDEFGHI"}
    seen = {"KKKKKKKK"}
    kept = accept(["ACDEFGHI", "KKKKKKKK", "ACDEFGHK", "ACDEFGHK", "bad", "X" * 12], known, seen, 10)
    assert kept == ["ACDEFGHK"]


def test_split_order_independent_and_disjoint():
    sequences = ["ACDEFGHI", "KKKKKKKK", "ACDEFGHK"]
    assert split(sequences) == split(list(reversed(sequences)) + sequences)
    train, val = split(sequences)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(sequences)


@pytest.mark.parametrize("model_class", [PeptideLM, FlowPolicy])
def test_causality(model_class):
    seed_all(42)
    model = model_class().eval()
    x = torch.tensor([[20, 0, 1, 2, 3, 4, 5, 6]])
    y = x.clone()
    y[0, 5:] = 19
    with torch.no_grad():
        a, b = model(x, torch.tensor([8])), model(y, torch.tensor([8]))
    torch.testing.assert_close(a[:, :5], b[:, :5])


@pytest.mark.parametrize("model_class", [PeptideLM, FlowPolicy])
def test_seeded_sampling_and_gradients(model_class):
    seed_all(42)
    model = model_class().eval()
    seed_all(123)
    first = sample(model, [8, 12, 50], "cpu")
    seed_all(123)
    assert first == sample(model, [8, 12, 50], "cpu")
    assert all(valid(s) for s in first)
    assert list(map(len, first)) == [8, 12, 50]
    lp, _ = log_probability(model, first, "cpu")
    (-lp.mean()).backward()
    assert torch.isfinite(lp).all()
    assert model.head.weight.grad.abs().sum() > 0


def test_trajectory_balance_exact_small_tree():
    # Enumerated two-leaf tree: R=[1,3], Z=4, P_F=[1/4,3/4].
    reward = torch.tensor([1.0, 3.0])
    pf = reward / reward.sum()
    loss = trajectory_balance(reward.sum().log(), pf.log(), reward.log())
    assert loss.item() < 1e-12
    assert trajectory_balance(torch.tensor(0.0), pf.log(), reward.log()) > 0


def test_merge_exact_quotas_and_provenance(tmp_path, monkeypatch):
    import csv
    import itertools
    import json
    import generate
    from common import write_fasta, fasta
    monkeypatch.setattr(generate, "WORK", tmp_path)
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    write_fasta(tmp_path / "data/reference/antibacterial.fasta", ["YYYYYYYY"])
    sequences = ("".join(s) for s in itertools.product("ACDE", repeat=8))
    for name, quota in [("diffusion", 16667), ("lm", 16667), ("gflownet", 16666)]:
        write_fasta(tmp_path / f"artifacts/pools/{name}.fasta", list(itertools.islice(sequences, quota)))
    generate.merge(None)
    library = fasta(tmp_path / "artifacts/library/library.fasta")
    assert len(library) == len(set(library)) == 50000
    with (tmp_path / "artifacts/library/provenance.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["sequence"] for r in rows] == library
    report = json.loads((tmp_path / "artifacts/library/validation.json").read_text())
    assert report["exact_reference_overlap"] == 0
    assert report["gflownet"]["selected"] == 16666


def test_merge_refuses_short_pool(tmp_path, monkeypatch):
    import generate
    from common import write_fasta
    monkeypatch.setattr(generate, "WORK", tmp_path)
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    write_fasta(tmp_path / "data/reference/antibacterial.fasta", [])
    write_fasta(tmp_path / "artifacts/pools/diffusion.fasta", ["ACDEFGHI"])
    with pytest.raises(RuntimeError, match="Insufficient unique diffusion"):
        generate.merge(None)
    assert not (tmp_path / "artifacts/library/library.fasta").exists()


def test_ddim_known_denoiser_and_length_conditioning():
    from types import SimpleNamespace
    from ddim import ddim_sample
    class ExactDenoiser:
        num_timesteps, seq_length, embed_dim, self_condition = 1000, 4, 2, False
        betas = torch.zeros(1000)
        alphas_cumprod = torch.linspace(0.999, 0.01, 1000)
        unnormalize = staticmethod(lambda x: x)
        def model_predictions(self, x, t, design_len, x_self_cond, clip_x_start):
            assert design_len == 12 and not clip_x_start and x_self_cond is None
            a = self.alphas_cumprod[t][:, None, None]
            clean = torch.full_like(x, 0.25)
            noise = (x - a.sqrt() * clean) / (1 - a).sqrt()
            return SimpleNamespace(pred_x_start=clean, pred_noise=noise)
    out = ddim_sample(ExactDenoiser(), 3, 12, steps=100)
    torch.testing.assert_close(out, torch.full((3, 4, 2), 0.25))
    with pytest.raises(ValueError):
        ddim_sample(ExactDenoiser(), 3, 12, steps=0)
