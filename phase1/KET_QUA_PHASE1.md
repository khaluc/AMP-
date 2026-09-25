# Kết quả Phase 1

Đã hoàn thành bản triển khai thử nghiệm ba nhánh sinh và thư viện 50.000 chuỗi.
Thời điểm ghi báo cáo (UTC): 2026-09-19T16:48:31.808495+00:00.

| Hạng mục | Kết quả |
|---|---|
| AMP-Diffusion | Checkpoint công bố + **DDIM 100 bước, eta=0**; chọn 16.667 chuỗi |
| Language model | Transformer tự hồi quy nhỏ, tiền huấn luyện UniProt rồi fine-tune AMP; chọn 16.667 chuỗi |
| GFlowNet | GRU riêng, warm-start có giám sát + 1.000 bước trajectory balance; chọn 16.666 chuỗi |
| Thư viện | Đúng 50.000 chuỗi duy nhất, 20 amino acid chuẩn, dài 8–50, không trùng hoàn toàn tham chiếu |
| Tái lập | Kiểm tra cùng seed và tái chạy hai batch đầu, gồm ranh giới resume, cho cả ba nhánh |

## Các tệp nên mở

- `artifacts/library/library.fasta`: thư viện hợp nhất.
- `artifacts/library/provenance.csv`: nguồn generator của từng chuỗi.
- `artifacts/library/independent_audit.json`: kiểm tra độc lập và thống kê mô tả.
- `artifacts/production_replay_checks.json`: kết quả tái lập prefix và resume.
- `artifacts/checkpoints/`: trọng số LM/GFlowNet, lịch sử huấn luyện và metadata.
- `artifacts/data/`: dữ liệu đã chuẩn bị, nguồn và báo cáo trùng lặp.
- `README.md`: phương pháp, giới hạn và lệnh chạy lại.

## Những điều kết quả này chưa chứng minh

- Chưa có đo hoạt tính, MIC dự đoán được hiệu chuẩn, điểm Aggregation Score chính thức hoặc top-100.
- LM là mô hình nhỏ của dự án, **không phải AMP-GPT đã công bố**.
- DDIM là biến thể suy luận được chọn để giảm thời gian chạy, chưa chứng minh tương đương sampler gốc về hoạt tính.
- GFlowNet tối ưu reward đại diện: prior từ LM, phân bố tính chất lý hóa và tránh trùng. Chưa có bảo đảm hội tụ chính xác hay ưu thế đa dạng.
- Ba nhánh khác kiến trúc; LM và GFlowNet vẫn chia sẻ dữ liệu và reward teacher, không độc lập thống kê.
- Đã kiểm tra tái lập các prefix trên cùng máy; chưa chạy lại toàn bộ 50.000 chuỗi hoặc kiểm chứng byte-identical giữa các GPU.
- Tập validation hiện chia theo chuỗi, chưa chia theo họ tương đồng/lab; không dùng để tuyên bố khả năng khái quát ngoài phân phối.
- Audit phát hiện 5 chuỗi từ nhánh LM trùng tập train AMP nhưng không trùng reference chính thức. Chúng chưa vi phạm phép kiểm tra exact-reference hiện tại; cần loại khỏi danh sách ứng viên mới ở bước lựa chọn sau.
- Có **4126** chuỗi MIC để riêng trùng dữ liệu train của checkpoint diffusion công bố. Phase 2 phải xử lý nguồn dữ liệu/assay và tính độc lập trước khi calibration.

SHA-256 của thư viện: `8402a234d5a51059d7ad3c6e547ad80687bfec220d77a449dfa89b35651c8405`.

Phase 2 chưa bắt đầu. Mục tiêu tiếp theo là audit dữ liệu MIC và thiết kế đánh giá độc lập, trước khi đưa ra bất kỳ bảo đảm conformal nào.
