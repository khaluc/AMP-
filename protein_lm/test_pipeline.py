import torch
from core import tokenizer, prefix, encode, batch, AA
from sample import token_table


def test_conditioning_and_padding_do_not_mask_real_eos():
    tok = tokenizer()
    a = encode(tok, "ACDEFGHIKLMN", "AMP")
    b = encode(tok, "KKKKKKKK", "NONAMP")
    assert prefix(tok, "AMP") != prefix(tok, "NONAMP")
    packed = batch([a, b], tok, device="cpu")
    for i, (ids, labels) in enumerate([a, b]):
        assert packed["labels"][i, len(ids)-1].item() == tok.eos_token_id
        assert packed["labels"][i, len(ids):].eq(-100).all()
        assert packed["attention_mask"][i, :len(ids)].eq(1).all()
        assert packed["attention_mask"][i, len(ids):].eq(0).all()
        assert labels[:len(prefix(tok, ["AMP", "NONAMP"][i]))] == [-100] * len(prefix(tok, ["AMP", "NONAMP"][i]))


def test_bpe_lengths_are_amino_acids_not_token_counts():
    tok = tokenizer()
    table = token_table(tok)
    assert all(not s or set(s) <= set(AA) for s in table)
    assert all(aa in table for aa in AA)
    assert table[tok.eos_token_id] == ""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    ids = tok.encode(sequence, add_special_tokens=False)
    assert "".join(table[i] for i in ids) == sequence
    assert sum(len(table[i]) for i in ids) == len(sequence)


def test_lora_changes_only_adapter_and_restores(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.manual_seed(42)
    config = GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=1, n_head=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    base = GPT2LMHeadModel(config)
    original = {k: v.clone() for k, v in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["c_attn", "c_proj"], fan_in_fan_out=True))
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    x = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(x, labels=x).loss.backward()
    optimizer.step()
    model.eval()
    expected = model(x).logits.detach()
    model.save_pretrained(tmp_path, safe_serialization=True)
    fresh = GPT2LMHeadModel(config)
    fresh.load_state_dict(original)
    restored = PeftModel.from_pretrained(fresh, tmp_path).eval()
    torch.testing.assert_close(expected, restored(x).logits)
