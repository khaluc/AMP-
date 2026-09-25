> Historical local audit, before GitHub publication. Current packaging, license scope and fresh-environment checks: [../README.md](../README.md), [DATA_AND_LICENSES.md](../DATA_AND_LICENSES.md), [VALIDATION.md](../VALIDATION.md).

# Trạng thái dự án AMP — 25/09/2026

Đã hoàn thiện bộ chạy, kiểm tra và báo cáo của phần tính toán hiện có. Một số mục tiêu khoa học vẫn chưa đạt.

| Hạng mục | Kết quả |
|---|---|
| Phase 1 | 50.000 chuỗi duy nhất, đủ quota ba nhánh; loại trùng với reference và dữ liệu đã chỉ định |
| Tái lập checkpoint | Đã sinh lại đủ cả ba pool từ checkpoint trên GPU; 50.000 chuỗi và top100 khớp byte bản gốc |
| Phase 2 | Tính lại đủ 50.000 dự đoán bằng mô hình đóng băng và feature cache có hash |
| Mục tiêu coverage | Test hai phía 276/358 = 77,095%; chưa đạt 90% |
| Phase 3 | Dựng lại đúng danh sách và thứ tự 100 ứng viên từ embedding, centroid và hàm greedy |
| Kiểm thử | 28 tests; trạng thái pass: True |
| Validator gốc | Kiểm tra chuỗi local pass: True |
| seqme | Đánh giá nội bộ đã chạy: True; không tự đặt Aggregation Score chính thức |
| MARTINI/MD | Chưa chạy; thiếu đầu vào cấu trúc, topology và hệ màng được xác định/kiểm tra |

## Chạy lại

Tại thư mục AMP:

```powershell
uv run --project completion python -m amp_completion.pipeline
```

Mặc định sinh lại từ checkpoint. Có thể dùng `--mode cached --output generate-cached` để kiểm tra nhanh từ pool đã lưu;
chế độ cached không phải full model replay. Entry point `generate` đã đóng gói nhưng Windows Application Control trên
máy này chặn launcher `.exe`; hai lần gọi module Python ở chế độ cached đã được đối chiếu và khớp từng byte.
Không thay đổi chính sách bảo mật Windows.

## Bằng chứng

- [Audit độc lập](artifacts/audit.json), [hiệu chỉnh](artifacts/calibration_audit.json).
- [seqme](artifacts/seqme_metrics.json), [validator gốc](artifacts/organizer_validation.json).
- [Trạng thái có cấu trúc](artifacts/status.json), [hướng dẫn](README.md), [phương pháp](METHODS.md).
- [Thiếu đầu vào MD](MD_READINESS.md), [nguồn và phạm vi giấy phép](DATA_AND_LICENSES.md).

## Phần chưa hoàn tất

1. Chứng minh coverage 90% trên dữ liệu đánh giá độc lập; không chỉnh ngưỡng theo test đã xem.
2. Mô phỏng MARTINI/MD sau khi xác định hệ và các đầu vào có cơ sở; chưa gán nhãn cơ chế cho cụm.
3. Kiểm tra trên một máy/GPU khác và xác nhận hoạt tính/độc tính bằng thực nghiệm.
4. Aggregation Score chính thức còn thiếu cấu hình công bố trong starter kit được kiểm tra.
5. Chưa xuất bản hoặc nộp bài; quyền phân phối các dữ liệu gốc còn cần xác định trước khi công khai.

Các báo cáo cũ được giữ nguyên để bảo toàn manifest. Dòng “Phase 3 chưa chạy” trong báo cáo Phase 2 cũ là lịch sử;
Phase 3 đã có kết quả lựa chọn riêng. Đây là dự án AMP, không phải docking EGFR/KRAS.
