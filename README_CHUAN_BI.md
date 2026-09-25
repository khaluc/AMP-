# Tài nguyên chuẩn bị AMP Challenge

Cập nhật: đã hoàn thành bản thử nghiệm Phase 1 với thư viện 50.000 chuỗi và môi trường GPU trong `phase1/`. Xem [kết quả Phase 1](phase1/KET_QUA_PHASE1.md) và [hướng dẫn chạy lại](phase1/README.md). Các ghi chú bên dưới mô tả trạng thái tại thời điểm tải tài nguyên ban đầu.

Đã tải từ các nguồn chính thức ngày 19/09/2026. Đây là bộ tài nguyên chuẩn bị; chưa cài môi trường Python, chưa huấn luyện hoặc chạy tạo bài nộp.

| Thư mục | Nội dung |
|---|---|
| `competition-template/` | Repository mẫu chính thức, PDF cuộc thi, validator, mã ví dụ, trọng số ví dụ, `uv.lock` và giấy phép |
| `data/reference/antibacterial.fasta` | Bản sao dữ liệu tham chiếu chính thức để kiểm tra trùng và độ tương đồng; không phải tập train mặc định |
| `starter-kits/hydramp/` | Bộ khởi đầu HydrAMP, trọng số, dữ liệu huấn luyện, dữ liệu MIC/hemolysis thực nghiệm có sẵn và tài liệu nguồn |
| `starter-kits/ampdiffusion/` | Bộ khởi đầu AMP-Diffusion, dữ liệu train, trọng số diffusion và 8 mô hình APEX qua Git LFS, dữ liệu thực nghiệm và tài liệu |
| `tools/seqme/` | Mã nguồn, tài liệu và kiểm thử của thư viện đánh giá seqme |
| `tools/hydramp-source/` | Mã nguồn HydrAMP tại đúng commit được starter kit chỉ định |
| `models/torch/hub/checkpoints/` | Trọng số ESM2-8M bổ sung cho AMP-Diffusion, xem manifest để biết các tệp tải thành công |
| `docs/` | Nội dung bạn cung cấp, HTML trang chủ, danh sách nguồn/commit và manifest SHA-256 |

## Phần chưa lấy được hoặc cần chuẩn bị tiếp

- **Kaggle:** API liệt kê dữ liệu cuộc thi trả HTTP 401 Unauthorized. Máy chưa có cấu hình xác thực Kaggle được phát hiện. Chưa thể xác nhận hoặc tải các tệp bổ sung trong tab Data. Cần đăng nhập Kaggle, tham gia/chấp nhận quy định cuộc thi nếu được yêu cầu, rồi cấu hình API token tại máy. Không gửi token vào hội thoại.
- **Môi trường chạy:** máy có Python 3.14 và uv, nhưng các dự án dùng phiên bản khác nhau: template ghim Python 3.11; HydrAMP yêu cầu Python 3.8; AMP-Diffusion yêu cầu Python 3.10. Các dependency được mô tả trong `pyproject.toml`/`uv.lock`, chưa được cài hoặc tải thành bộ wheel offline.
- **Điểm chính thức:** tải seqme không đồng nghĩa đã có đầy đủ cấu hình Aggregation Score của ban tổ chức. README các starter kit còn ghi phần `metrics/` sẽ được bổ sung; cần kiểm tra công bố/Kaggle trước khi dùng điểm nội bộ để so sánh.
- **Nguồn dữ liệu ngoài:** đã lấy các tập train được phân phối trong hai starter kit. Không coi toàn bộ DBAASP/APD/dbAMP/Peptipedia là dữ liệu bắt buộc phải tải; mô tả cuộc thi cho phép lựa chọn dữ liệu. Thông tin nguồn và điều khoản nằm trong `data/README.md` của từng kit.

## Lưu ý khi bắt đầu

- Giữ nguyên các repository đã tải làm bản tham chiếu; chọn phương pháp rồi tạo dự án bài nộp riêng.
- Thiết lập `TORCH_HOME` trỏ tới `D:\Blue AI\AMP\models\torch` khi dùng ESM2 đã tải trong thư mục này.
- Template dùng entry point `generate`, còn các starter kit dùng `generate_broad_spectrum`. Cần đối chiếu quy định mới nhất trước khi nộp.
- Bài nộp cần thư viện 50.000 chuỗi và top 100 có xếp hạng, tài liệu phương pháp, nguồn dữ liệu và mã/trọng số tái lập. Đây là đầu ra phải tạo, không phải tệp có sẵn để tải.
- Không suy ra hạn chót từ dòng “12 days to go” trong bản dán; kiểm tra trực tiếp trên Kaggle.

## Nguồn

- https://github.com/szczurek-lab/amp-challenge-2027
- https://github.com/szczurek-lab/hydramp-starter-kit
- https://github.com/szczurek-lab/ampdiffusion-starter-kit
- https://github.com/szczurek-lab/seqme
- https://github.com/szczurek-lab/hydramp
- https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py
- https://www.kaggle.com/competitions/amp-challenge/data

Chạy `python scripts/inventory.py` từ thư mục AMP để cập nhật manifest kích thước và SHA-256. Manifest giúp đối chiếu các tệp tải về, không chứng nhận chất lượng mô hình hoặc khả năng chạy của môi trường.
