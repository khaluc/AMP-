> Historical local audit, before GitHub publication. Current packaging, license scope and fresh-environment checks: [../README.md](../README.md), [DATA_AND_LICENSES.md](../DATA_AND_LICENSES.md), [VALIDATION.md](../VALIDATION.md).

# Hoàn thiện và tái lập dự án AMP

Báo cáo tổng hợp trong dự án: `completion/KET_QUA.md`; bằng chứng: `completion/artifacts/`.

Đây là bộ chạy và kiểm tra cho dự án peptide kháng khuẩn hiện có, không phải pipeline docking EGFR.
Ba nhánh sinh là AMP-Diffusion DDIM100, ProtGPT2-LoRA và GFlowNet GRU đa mục tiêu.

## Chạy lại từ checkpoint

Từ thư mục gốc AMP, dùng hai môi trường đã khóa trong `phase1/` và `protein_lm/`.
Nếu chưa có môi trường, bộ chạy tự gọi uv sync theo lock file. Có thể chủ động tạo trước bằng:

```powershell
uv sync --project phase1 --locked
uv sync --project protein_lm --locked
```

Chạy trên GPU CUDA, giữ batch size và seed gốc:

```powershell
uv run --project completion generate
```

Trên máy này, Windows Application Control chặn launcher `generate.exe`, kể cả khi chạy ngoài sandbox.
Cách gọi module Python đã được kiểm tra:

```powershell
uv run --project completion python -m amp_completion.pipeline
```

Thư mục mặc định là `generate/` trong thư mục hiện tại. Có thể đặt `--output` tới một thư mục mới.
Chạy lại cùng lệnh trên đầu ra hoàn chỉnh sẽ tạo thư mục replay riêng và so sánh byte của cả hai FASTA.
Đầu ra không có dấu xác nhận của bộ chạy hoặc đã bị sửa sẽ được giữ nguyên và lệnh báo lỗi.

Mặc định `--mode checkpoints` thực sự chạy lại inference của cả ba mô hình. Sau đó bộ kiểm tra
dựng thư viện từ các pool mới, tính lại 50.000 dự đoán bằng regressor đóng băng từ feature cache đã kiểm tra hash,
tính lại quantile và dựng lại portfolio từ embedding, centroid và hàm greedy.
Không huấn luyện lại mô hình, không thay seed, không thay ngưỡng theo kết quả test.
Chỉ hỗ trợ tái lập bản phát hành đã đóng băng; thư viện mới không khớp sẽ bị từ chối và cần pipeline chấm điểm mới.

Kiểm tra nhanh từ pool đã lưu:

```powershell
uv run --project completion python -m amp_completion.pipeline --mode cached --output generate-cached
```

`cached` chỉ là dựng lại artifact, không phải sinh lại từ checkpoint. Các báo cáo phân biệt rõ hai chế độ.

## Đánh giá seqme và validator gốc

Các dependency bổ sung được ghim phiên bản và cài riêng, không nâng cấp môi trường huấn luyện:

```powershell
uv pip install --python phase1/.venv/Scripts/python.exe --target completion/.eval-deps --no-deps -r completion/requirements-evaluation.txt
$env:PYTHONPATH = "$PWD/completion/src;$PWD/completion/.eval-deps"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = "$PWD/completion/artifacts/matplotlib"
$env:NUMBA_CACHE_DIR = "$PWD/completion/artifacts/numba-cache"
phase1/.venv/Scripts/python.exe -m amp_completion.evaluate --root . --output generate
```

Count, uniqueness, exact novelty và length được tính trên toàn bộ các nhóm. Diversity dùng mẫu cố định tối đa
2.000 chuỗi mỗi nhóm, 10 đối tác mỗi chuỗi; top100 dùng mọi cặp. Đây là chỉ số nội bộ, không phải Aggregation Score chính thức.
`evaluate` gọi trực tiếp các hàm kiểm tra chuỗi của validator gốc, bao gồm ngưỡng Levenshtein ratio 80%.
Không thực hiện clone/push hoặc nộp bài trực tuyến.

## Đầu ra

- `library.fasta`: 50.000 chuỗi theo đúng thứ tự và ID gốc.
- `top.fasta`, `top.csv`: 100 ứng viên, thứ tự greedy xây dựng portfolio, không phải thứ tự potency đơn lẻ.
- `audit.json`: kiểm tra nguồn, hash, dự đoán, full pool replay và giới hạn.
- `calibration_audit.json`: kết quả hiệu chỉnh và coverage của từng split.
- `seqme_metrics.json`, `seqme_metrics.csv`, `organizer_validation.json`: đánh giá độc lập.
- `logs/`: stdout/stderr của từng bước; `run.json`: trạng thái thành công hoặc lỗi.

## Giới hạn khoa học còn tồn tại

Coverage hai phía test là 276/358 = 77,095%, mục tiêu danh nghĩa 90% chưa đạt. Split `validation` được dùng khi
refit mô hình cuối cùng, nên residual của split này trong audit là in-sample; không dùng nó làm đánh giá độc lập.
Không chỉnh quantile theo test. Cần tập đánh giá độc lập mới cho lần phát triển mô hình tiếp theo.

MARTINI/MD chưa có cấu trúc, topology và hệ màng được xác định/kiểm tra cho 12 đại diện. Xem `MD_READINESS.md`.
Không suy ra cơ chế tác động từ clustering hoặc tự điền nhãn MoA.
Tái lập trên máy hiện tại không chứng minh tính giống hệt trên GPU/máy khác; hoạt tính và độc tính vẫn chưa được xác nhận thực nghiệm.

## Nguồn đối chiếu

- [Yêu cầu cuộc thi](https://github.com/szczurek-lab/amp-challenge-2027), bản local ghi commit trong `docs/sources.json`.
- [seqme](https://github.com/szczurek-lab/seqme), dùng phiên bản 0.5.1.
- [Metrics của starter kit hiện là placeholder](https://github.com/szczurek-lab/ampdiffusion-starter-kit/tree/main/metrics), kiểm tra ngày 25/09/2026.
- Xem `METHODS.md`, `DATA_AND_LICENSES.md` và các manifest đi kèm để biết phạm vi bản đóng gói.
