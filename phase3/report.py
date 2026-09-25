from pathlib import Path
import json,hashlib
import pandas as pd
root=Path(__file__).resolve().parent;out=root/'artifacts'
d=json.loads((out/'result.json').read_text()); a=d['baseline'];b=d['portfolio']
t=pd.read_csv(out/'top100_portfolio_exploratory.csv')
assert len(t)==t.sequence.nunique()==100
lines=(out/'top100_portfolio_exploratory.fasta').read_text().splitlines()
assert lines[1::2]==t.sequence.tolist()
assert [s[1:] for s in lines[::2]]==t.id.tolist()
comparison=pd.DataFrame([{'method':'phase2_MIC_ranked',**a},{'method':'phase3_portfolio',**b}])
comparison.to_csv(out/'comparison.csv',index=False)
report=f'''# Kết quả Phase 3 — chọn portfolio thử nghiệm

Đã chọn đủ **100 peptide** bằng greedy facility-location kết hợp chất lượng MIC dự đoán.
Giữ nguyên điều kiện Phase 2: upper bound thử nghiệm <=16 µM và similarity <=80%.
Không chia quota cho ba generator. Không thay đổi top100 cũ.

## Kết quả so với top100 theo MIC

| Chỉ số | Phase 2 | Phase 3 |
| --- | --- | --- |
| Số chuỗi | {a['n']} | {b['n']} |
| Số cụm hình học được đại diện | {a['hard_clusters']} | {b['hard_clusters']} |
| Facility coverage (điểm kernel, không phải phần trăm MoA) | {a['facility_coverage']:.4f} | {b['facility_coverage']:.4f} |
| Tổng mục tiêu đã định nghĩa | {a['objective']:.4f} | {b['objective']:.4f} |
| Khoảng cách Indel trung bình giữa các cặp | {a['mean_pairwise_indel_distance']:.4f} | {b['mean_pairwise_indel_distance']:.4f} |
| Số cặp similarity >80% trong top100 | {a['pairs_above_80pct_similarity']} | {b['pairs_above_80pct_similarity']} |
| Median cận trên MIC thử nghiệm (µM) | {a['median_upper_uM']:.3f} | {b['median_upper_uM']:.3f} |
| Max cận trên MIC thử nghiệm (µM) | {a['max_upper_uM']:.3f} | {b['max_upper_uM']:.3f} |
| Trung bình số cụm trong 25 mẫu ngẫu nhiên | {a['random25_distinct_clusters_mean']:.2f} | {b['random25_distinct_clusters_mean']:.2f} |

Phân bố generator Phase 2: **{a['generators']}**.
Phân bố generator Phase 3: **{b['generators']}**.
Hai danh sách trùng **{d['shared_with_baseline']}/100** chuỗi.

## Giới hạn độ phủ

Phân toàn bộ 50.000 chuỗi thành 100 cụm hình học bằng ESM2-8M + 10 đặc trưng lý hóa.
Nhưng 167 ứng viên đạt bộ lọc chỉ nằm trong **{d['eligible_hard_clusters']} cụm**;
**{d['empty_of_eligible_clusters']} cụm không có ứng viên đạt điều kiện**.
Vì thế không thể chọn một peptide từ mỗi cụm mà vẫn giữ bộ lọc hiện tại.
Phương pháp dùng độ phủ mềm qua khoảng cách tới 100 centroid, không tự nới MIC lên64 µM.

Simulate 2.000 lượt lấy 25 mẫu không hoàn lại; các kết quả chỉ mô tả phân bố cụm.
P05/P50/P95 số cụm trong25 mẫu: Phase2 {a['random25_distinct_clusters_p05_p50_p95']};
Phase3 {b['random25_distinct_clusters_p05_p50_p95']}.
Đây không phải mô phỏng hit-rate và không phải xác suất phủ các cơ chế tác động.

## Kiểm tra và cách diễn giải

- Đủ100 chuỗi duy nhất, thuộc thư viện gốc, ký tự/độ dài hợp lệ.
- Giữ upper_MIC<=16µM; đối chiếu scalar similarity với toàn bộ reference cho từng ứng viên.
- Kiểm tra lại greedy cùng đầu vào cho kết quả giống nhau; unit tests kiểm tra diminishing
  returns, biên approximation trên bài toán nhỏ giải chính xác, tie-break và marginal gains.
- Cận approximation 1−1/e chỉ áp dụng cho mục tiêu submodular đã định nghĩa.
- **Coverage MIC test vẫn77,1%; không có bảo đảm hit-rate hoặc MoA.**
- Chưa chạy MARTINI, chưa gán nhãn carpet/pore/intracellular. Đây là phần tùy chọn còn lại.
  Có danh sách {d['cluster_representatives']} đại diện để phục vụ nghiên cứu tiếp theo,
  tất cả được ghi cơ chế là chưa xác định.

## Tệp đầu ra

- `artifacts/top100_portfolio_exploratory.csv`: portfolio, nguồn sinh, cụm, MIC và gain.
- `artifacts/top100_portfolio_exploratory.fasta`:100 chuỗi, giữ ID thư viện gốc.
- `artifacts/library_clusters.csv`: nhãn cụm cho50.000 chuỗi.
- `artifacts/cluster_summary.csv`: số chuỗi toàn thư viện/đạt lọc/được chọn mỗi cụm.
- `artifacts/greedy_trace.json`, `selection_inputs.npz`: đầu vào và diễn tiến lựa chọn.
- `artifacts/result.json`, `comparison.csv`, `manifest.json`: bằng chứng và fingerprint.
- `README.md`: công thức, tham số cố định và lệnh tái chạy.

Thứ tự FASTA là thứ tự thêm vào portfolio theo marginal gain, không phải thứ hạng hoạt tính.
'''
(root/'KET_QUA.md').write_text(report,encoding='utf8')
print(json.dumps({'generators':b['generators'],'clusters':b['hard_clusters'],'baseline_clusters':a['hard_clusters'],'eligible_clusters':d['eligible_hard_clusters'],'shared':d['shared_with_baseline'],'objective_gain':b['objective']-a['objective']},indent=2))
