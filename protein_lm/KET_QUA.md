# Kết quả nhánh 2: ProtGPT2 + LoRA

Đã fine-tune checkpoint `nferruz/ProtGPT2` bằng LoRA rank 16, hoàn thành 3 epoch.
Base có 774.030.080 tham số; số tham số LoRA được huấn luyện: 8,110,080.
Đây là ProtGPT2-LoRA, không phải bản tái lập AMP-GPT công bố hay ESM-2 causal.

## Kiểm tra huấn luyện

AMP validation token NLL trước fine-tune: **9.5946**.
NLL thấp hơn là tốt hơn cho dự đoán token, không chứng minh hoạt tính kháng khuẩn.

| Epoch | AMP token NLL | NONAMP token NLL |
| --- | --- | --- |
| 1 | 6.7833 | 7.3496 |
| 2 | 6.3853 | 7.2363 |
| 3 | 6.1636 | 7.2010 |

Chọn adapter có AMP validation NLL thấp nhất: **6.1636**.
PyTorch peak reserved của lượt train hoàn tất: **5006 MiB**;
lượt train trước khi bị ngắt từng ghi nhận 4464 MiB. Đây không phải tổng VRAM toàn hệ thống.
Batch 64, tích lũy gradient 4, BF16 base đóng băng, adapter FP32, gradient checkpointing.

## Đầu ra đã kiểm tra

- Pool mới: **17,536** chuỗi duy nhất từ **17,920** lượt sinh;
  độ dài 8–50 amino acid, chỉ 20 ký tự chuẩn.
- Không trùng nguyên chuỗi với reference chính thức hoặc dữ liệu fine-tune/validation.
- Thư viện v2: 50.000 chuỗi duy nhất, gồm 16.667 diffusion DDIM100,
  16.667 ProtGPT2-LoRA và 16.666 GFlowNet baseline.
- Replay cùng seed và tiếp tục sampling đã đối chiếu đúng 256 chuỗi đầu;
  chưa replay toàn bộ thư viện hoặc kiểm tra chéo phần cứng.
- Peak reserved khi sinh: 3308 MiB.
- SHA-256 thư viện: `ba2a397d4d4d4f61e27f7016a6b1565c56f1362bc0ccdfb05691b3c551225983`.

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
