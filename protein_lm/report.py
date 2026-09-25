"""Summarize completed runs from saved evidence; never infer success from logs alone."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase1"))
from common import fasta, sha, dump, valid

out = ROOT / "protein_lm/artifacts"
def read(name):
    return json.loads((out / name).read_text(encoding="utf-8"))

training = read("training/complete.json")
baseline = read("training/base_validation.json")
pool = read("pool/protgpt2_lora.progress.json")
audit = read("library_v2/audit.json")
replay = read("replay_checks.json")
assert training["status"] == "completed"
assert len(training["history"]) == 3
assert training["adapter_sha256"] == sha(out / "adapter/adapter_model.safetensors")
assert pool["settings"]["adapter_sha256"] == training["adapter_sha256"]
assert pool["sha256"] == sha(out / "pool/protgpt2_lora.fasta")
assert audit["library_sha256"] == sha(out / "library_v2/library.fasta")
sequences = fasta(out / "pool/protgpt2_lora.fasta")
assert len(sequences) == len(set(sequences)) == pool["accepted"] >= 17500
assert all(valid(s) for s in sequences)
assert replay["same_seed_production_prefix"] and replay["resume_prefix"]
summary = {
    "model": "nferruz/ProtGPT2 + LoRA r16",
    "training": training,
    "baseline_validation": baseline,
    "pool_count": len(sequences),
    "pool_attempts": pool["attempts"],
    "pool_acceptance_fraction": len(sequences) / pool["attempts"],
    "length_min": min(map(len, sequences)),
    "length_max": max(map(len, sequences)),
    "length_mean": sum(map(len, sequences)) / len(sequences),
    "generation_peak_reserved_mib": pool["peak_reserved_mib"],
    "library_audit": audit,
    "replay": replay,
}
dump(out / "result.json", summary)
rows = "\n".join(f"| {r['epoch']} | {r['AMP_token_nll']:.4f} | {r['NONAMP_token_nll']:.4f} |" for r in training["history"])
report = f"""# Kết quả nhánh 2: ProtGPT2 + LoRA

Đã fine-tune checkpoint `nferruz/ProtGPT2` bằng LoRA rank 16, hoàn thành 3 epoch.
Base có 774.030.080 tham số; số tham số LoRA được huấn luyện: {training['trainable_parameters']:,}.
Đây là ProtGPT2-LoRA, không phải bản tái lập AMP-GPT công bố hay ESM-2 causal.

## Kiểm tra huấn luyện

AMP validation token NLL trước fine-tune: **{baseline['AMP_token_nll']:.4f}**.
NLL thấp hơn là tốt hơn cho dự đoán token, không chứng minh hoạt tính kháng khuẩn.

| Epoch | AMP token NLL | NONAMP token NLL |
| --- | --- | --- |
{rows}

Chọn adapter có AMP validation NLL thấp nhất: **{training['best_AMP_token_nll']:.4f}**.
PyTorch peak reserved của lượt train hoàn tất: **{training['peak_reserved_mib']:.0f} MiB**;
lượt train trước khi bị ngắt từng ghi nhận 4464 MiB. Đây không phải tổng VRAM toàn hệ thống.
Batch 64, tích lũy gradient 4, BF16 base đóng băng, adapter FP32, gradient checkpointing.

## Đầu ra đã kiểm tra

- Pool mới: **{len(sequences):,}** chuỗi duy nhất từ **{pool['attempts']:,}** lượt sinh;
  độ dài {summary['length_min']}–{summary['length_max']} amino acid, chỉ 20 ký tự chuẩn.
- Không trùng nguyên chuỗi với reference chính thức hoặc dữ liệu fine-tune/validation.
- Thư viện v2: 50.000 chuỗi duy nhất, gồm 16.667 diffusion DDIM100,
  16.667 ProtGPT2-LoRA và 16.666 GFlowNet baseline.
- Replay cùng seed và tiếp tục sampling đã đối chiếu đúng {replay['checked_sequences']} chuỗi đầu;
  chưa replay toàn bộ thư viện hoặc kiểm tra chéo phần cứng.
- Peak reserved khi sinh: {pool['peak_reserved_mib']:.0f} MiB.
- SHA-256 thư viện: `{audit['library_sha256']}`.

Tệp chính: `artifacts/adapter/`, `artifacts/pool/protgpt2_lora.fasta`,
`artifacts/library_v2/library.fasta`, `artifacts/library_v2/provenance.csv`,
`artifacts/result.json`. Bản baseline trong `phase1/` được giữ nguyên.

## Giới hạn

UniProt controls là nhóm giả định âm tính, không phải peptide đã đo không có hoạt tính.
Nhóm này chỉ dài tới 25 aa nên có thể gây nhiễu nhãn theo độ dài.
Validation tách theo trình tự, chưa tách theo họ tương đồng hay phòng thí nghiệm.
Loại exact overlap không chứng minh novelty theo tương đồng.
Chưa đánh giá MIC, độc tính, top-100 hoặc conformal coverage.
GFlowNet vẫn dùng reward từ mô hình PeptideLM nhỏ cũ; chỉ nhánh 2 được nâng cấp ở đây.
"""
(ROOT / "protein_lm/KET_QUA.md").write_text(report, encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
