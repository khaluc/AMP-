"""Write a completion report only after the independent library audit and replay pass."""
from datetime import datetime, timezone
from common import WORK, dump, sha
import json


def read(name):
    return json.loads((WORK / "artifacts" / name).read_text(encoding="utf-8"))


audit = read("library/independent_audit.json")
replay = read("production_replay_checks.json")
data = read("data/independent_audit.json")
assert audit["structural_checks_passed"] and audit["count"] == 50000
assert audit["library_sha256"] == sha(WORK / "artifacts/library/library.fasta"), "Library changed after audit"
assert set(replay) == {"lm", "gflownet", "diffusion"}
assert all(v["first_batch_match"] and v["resumed_prefix_match"] for v in replay.values())
if (WORK / "artifacts/resume_event.json").exists():
    assert replay["diffusion"]["actual_interruption_boundary_match"]
for name in ["lm", "gflownet"]:
    metadata = read(f"checkpoints/{name}.metadata.json")
    assert metadata["checkpoint_sha256"] == sha(WORK / f"artifacts/checkpoints/{name}.pt")

manifest = []
for pattern in ["*.py", "pyproject.toml", "uv.lock", ".python-version", "README.md"]:
    for path in sorted(WORK.glob(pattern)):
        manifest.append({"path": path.relative_to(WORK).as_posix(), "sha256": sha(path)})
dump(WORK / "artifacts/code_manifest.json", manifest)
status = {"phase": 1, "status": "completed_experimental_baseline", "completed_utc": datetime.now(timezone.utc).isoformat(), "library_count": 50000, "library_sha256": audit["library_sha256"], "diffusion_variant": "DDIM eta=0, 100 steps", "phase2_started": False, "biological_activity_validated": False}
dump(WORK / "artifacts/status.json", status)
text = f"""# Kết quả Phase 1

Đã hoàn thành bản triển khai thử nghiệm ba nhánh sinh và thư viện 50.000 chuỗi.
Thời điểm ghi báo cáo (UTC): {status['completed_utc']}.

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
- Audit phát hiện {audit['generators']['lm']['exact_training_overlap']} chuỗi từ nhánh LM trùng tập train AMP nhưng không trùng reference chính thức. Chúng chưa vi phạm phép kiểm tra exact-reference hiện tại; cần loại khỏi danh sách ứng viên mới ở bước lựa chọn sau.
- Có **{data['published_diffusion_train_overlap_with_reserved_mic']}** chuỗi MIC để riêng trùng dữ liệu train của checkpoint diffusion công bố. Phase 2 phải xử lý nguồn dữ liệu/assay và tính độc lập trước khi calibration.

SHA-256 của thư viện: `{audit['library_sha256']}`.

Phase 2 chưa bắt đầu. Mục tiêu tiếp theo là audit dữ liệu MIC và thiết kế đánh giá độc lập, trước khi đưa ra bất kỳ bảo đảm conformal nào.
"""
(WORK / "KET_QUA_PHASE1.md").write_text(text, encoding="utf-8")
print(json.dumps(status, indent=2))
