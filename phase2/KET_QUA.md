# Kết quả hoàn thiện Phase 1 và chạy Phase 2

## Phase 1: đã hoàn thành ensemble triển khai

- AMP-Diffusion checkpoint gốc với biến thể DDIM100 đã được chọn trước đó.
- ProtGPT2 tiền huấn luyện + LoRA rank16, train đủ 3 epoch.
- GFlowNet GRU được nâng cấp reward đa mục tiêu và train thêm 1000 bước
  Trajectory Balance: MIC dự đoán + novelty + lực đẩy khỏi tập anchor cố định + tính chất
  lý hóa, cùng LM prior. Các trọng số reward cố định, không chọn theo test/calibration.
- Thư viện cuối: đúng **50.000 chuỗi duy nhất**, quota 16.667 / 16.667 / 16.666,
  dài 8–50 aa, không trùng nguyên chuỗi với reference, dữ liệu fine-tune LM hoặc MIC.
- Kiểm tra tái sinh cùng seed và resume GFlowNet khớp 256
  chuỗi đầu. Diffusion và ProtGPT2 đã có kiểm tra prefix riêng ở lượt trước.
- File: `phase1/artifacts/final/library.fasta` và `provenance.csv`.
- SHA-256: `36b147ebe114d5176305f1b9fa59ce8f2f8fef7799c493c377f6ebb17a191aa9`.

Đây là bản triển khai theo ba họ mô hình, không phải tái lập nguyên vẹn các bài báo.
ESM2-8M, GRU GFlowNet và ProtGPT2-LoRA là lựa chọn thực tế được ghi rõ.
Chưa chạy toàn bộ seqme Aggregation Score, chưa full replay 50.000 chuỗi trên máy khác,
chưa hoàn thiện đóng gói repo nộp bài/kiểm tra quyền phân phối dữ liệu.

## Phase 2: pipeline đã chạy xong; mục tiêu thống kê chưa đạt

Nguồn: DBAASP subset trong bản GRAMPA 2018, không phải tải mới toàn bộ DBAASP.
Sau lọc còn **2260 trình tự**, 1173 nhóm tương đồng.
Train/validation/calibration/test: {'train': 1310, 'calibration': 377, 'test': 358, 'validation': 215}.
Đích dự đoán: median log10 MIC của mỗi trình tự trên các assay E. coli có sẵn;
không đại diện riêng strain hoặc điều kiện thử nghiệm của cuộc thi.

ESM2-8M embedding + 10 đặc trưng lý hóa; chọn **random_forest** bằng validation.

| Chỉ số test | Kết quả |
| --- | --- |
| MAE log10 MIC | 0.6167 |
| MAE baseline median | 0.7873 |
| Spearman | 0.6078 |
| Coverage hai phía, đặt mục tiêu 90% | **77.09%** |
| Coverage cận trên một phía, đặt mục tiêu 90% | 84.36% |

Quantile hai phía: 0.9890 log10, tính đúng thứ hạng hữu hạn mẫu.
Giữ nguyên alpha=0,1 và MIC=16 µM; không điều chỉnh theo test hoặc số ứng viên đạt.
**Không được tuyên bố coverage 90% đã được xác thực hoặc hit-rate được bảo đảm.**
Có 368/377
chuỗi calibration trùng dữ liệu train diffusion công bố. Còn khác biệt phân phối,
phụ thuộc theo nhóm, điều kiện assay và lựa chọn thích nghi của ứng viên sinh.
Ngay cả 25 phép thử độc lập với p=0,9, xác suất >=22 thành công chỉ là
0.7636, không phải >=0,9.

## Xếp hạng thử nghiệm

- Đã chấm đủ 50000 chuỗi: `phase2/artifacts/ranked_library.csv`.
- Qua novelty theo đúng normalized Indel/Levenshtein.ratio của cuộc thi:
  **46881** chuỗi.
- Đồng thời upper bound thử nghiệm <=16 µM và similarity <=80%:
  **167** chuỗi.
- Trạng thái top100: `exploratory_candidates_written`.
- File đạt bộ lọc: `eligible_candidates.csv` / `.fasta`.
- Top-100 thử nghiệm: `top100_exploratory.csv` / `.fasta` (đã xuất trong lượt này).
- Nếu không đủ 100: chỉ có CSV diagnostic ghi rõ không đạt ngưỡng; không bù bằng
  ứng viên yếu hơn, không đổi alpha hoặc gọi đó là top100 đạt chuẩn.

Khoảng dự đoán và danh sách đạt bộ lọc vẫn chỉ là thử nghiệm; không phải xác nhận MIC,
an toàn tế bào, cơ chế hoặc hiệu quả thực nghiệm. Phase 3 chưa chạy.

## Tái lập và bằng chứng

`phase2/README.md` có nguồn, phương pháp, giới hạn và lệnh chạy.
`artifacts/verification.json` chứa kiểm tra độc lập split, quantile, model reload,
thư viện, scalar similarity và replay/resume. `artifacts/result.json` tổng hợp kết quả.
Các bản baseline cũ được giữ nguyên để đối chiếu.
