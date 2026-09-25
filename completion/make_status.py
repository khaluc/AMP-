"""Summarize existing evidence; never convert missing work into a passing status."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "artifacts")
args = parser.parse_args()
root, results = args.root.resolve(), args.results.resolve()
audit = json.loads((results / "audit.json").read_text(encoding="utf-8"))
cal = json.loads((results / "calibration_audit.json").read_text(encoding="utf-8"))
tests = []
for path in sorted(results.glob("tests-*.xml")):
    for suite in ET.parse(path).getroot().iter("testsuite"):
        tests.append({"report": path.name, "tests": int(suite.get("tests", 0)),
                      "failures": int(suite.get("failures", 0)), "errors": int(suite.get("errors", 0))})
checkpoint = results / "checkpoint-replay/run.json"
replay = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
full_replay = replay.get("status") == "computational_reconstruction_verified" and replay.get("model_replay") is True
md_path = results / "md_readiness.json"
md = json.loads(md_path.read_text(encoding="utf-8")) if md_path.exists() else {"simulation_executed": False}
seqme = (results / "seqme_metrics.json").exists()
organizer = results / "organizer_validation.json"
organizer_passed = organizer.exists() and json.loads(organizer.read_text(encoding="utf-8")).get("passed") is True
status = {"all_scientific_phases_complete": False,
    "library_count": audit["library_count"], "top_count": audit["top_count"],
    "full_checkpoint_replay_same_machine": full_replay,
    "phase2_statistical_target_met": cal["statistical_target_met"],
    "phase2_test_coverage": cal["by_split"]["test"]["coverage_two_sided"],
    "phase3_portfolio_recomputed_from_features": audit["portfolio_recomputed_from_features"],
    "tests": tests, "unit_tests_passed": bool(tests) and all(t["errors"] == t["failures"] == 0 for t in tests),
    "seqme_local_evaluated": seqme, "official_aggregation_score": None,
    "organizer_local_sequence_checks_passed": organizer_passed,
    "md_completed": md.get("simulation_executed", False),
    "console_launcher": "blocked_by_windows_application_control_in_this_session",
    "module_entrypoint": "cached_repeated_invocations_byte_identical",
    "cross_hardware_replay": False, "experimental_validation": False, "published_or_submitted": False}
(results / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
count = sum(t["tests"] for t in tests)
replay_text = "Đã sinh lại đủ cả ba pool từ checkpoint trên GPU; 50.000 chuỗi và top100 khớp byte bản gốc" if full_replay else "Chưa có bằng chứng hoàn thành full checkpoint replay"
report = f'''# Trạng thái dự án AMP — 25/09/2026

Đã hoàn thiện bộ chạy, kiểm tra và báo cáo của phần tính toán hiện có. Một số mục tiêu khoa học vẫn chưa đạt.

| Hạng mục | Kết quả |
|---|---|
| Phase 1 | 50.000 chuỗi duy nhất, đủ quota ba nhánh; loại trùng với reference và dữ liệu đã chỉ định |
| Tái lập checkpoint | {replay_text} |
| Phase 2 | Tính lại đủ 50.000 dự đoán bằng mô hình đóng băng và feature cache có hash |
| Mục tiêu coverage | Test hai phía 276/358 = 77,095%; chưa đạt 90% |
| Phase 3 | Dựng lại đúng danh sách và thứ tự 100 ứng viên từ embedding, centroid và hàm greedy |
| Kiểm thử | {count} tests; trạng thái pass: {status['unit_tests_passed']} |
| Validator gốc | Kiểm tra chuỗi local pass: {organizer_passed} |
| seqme | Đánh giá nội bộ đã chạy: {seqme}; không tự đặt Aggregation Score chính thức |
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
'''
(Path(__file__).resolve().parent / "KET_QUA.md").write_text(report, encoding="utf-8")
print(json.dumps(status, indent=2, ensure_ascii=False))
