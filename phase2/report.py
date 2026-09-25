from core import *
import pandas as pd
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
data=read(OUT/'data_audit.json');cal=read(OUT/'calibration.json');selection=read(OUT/'selection_audit.json');checks=read(OUT/'verification.json')
flow=read(ROOT/'phase1/artifacts/multiobjective/metadata.json');library=read(ROOT/'phase1/artifacts/final/audit.json')
assert flow['status']=='completed' and library['sha256']==selection['library_sha256']
assert cal['model_sha256']==flow['oracle_sha256']==selection['model_sha256']
ranked=pd.read_csv(OUT/'ranked_library.csv')
if selection['eligible_at_16uM']>=100:
    top=pd.read_csv(OUT/'top100_exploratory.csv')
    assert len(top)==top.sequence.nunique()==100
    assert set(top.sequence)<=set(ranked.sequence)
    assert (top.upper_uM<=16).all() and (top.max_reference_similarity<=.8).all()
    assert fasta(OUT/'top100_exploratory.fasta')==top.sequence.tolist()
    eligible=pd.read_csv(OUT/'eligible_candidates.csv')
    assert top.sequence.tolist()==eligible.head(100).sequence.tolist()
    checks['top100_membership_uniqueness_thresholds_order_fasta_match']=True
    dump(OUT/'verification.json',checks)
seq=ranked.sort_values('id').sequence.tolist()
rng=np.random.default_rng(42)
from rapidfuzz.distance import Indel
diversity={}
for name,g in ranked.groupby('generator'):
    sample=g.sequence.to_numpy()[rng.choice(len(g),min(500,len(g)),replace=False)].tolist()
    matrix=process.cdist(sample,sample,scorer=Indel.normalized_similarity,workers=4,dtype=np.float64)
    diversity[name]={'sample_n':len(sample),'mean_pairwise_indel_distance':float((1-matrix)[np.triu_indices(len(sample),1)].mean()),
       'mean_length':float(g.sequence.str.len().mean()),'novelty_pass_fraction':float(g.passes_reference_80pct.mean())}
dump(ROOT/'phase1/artifacts/final/diversity_audit.json',diversity)
summary={'phase1':'operational_ensemble_completed','phase2':'exploratory_pipeline_completed_statistical_target_failed' if not checks['statistical_target_met'] else 'exploratory_pipeline_completed',
 'data':data,'calibration':cal,'selection':selection,'checks':checks,'diversity':diversity}
dump(OUT/'result.json',summary)
tracked=list(Path(__file__).parent.glob('*.py'))+[ROOT/'phase1/uv.lock',ROOT/'phase1/networks.py',
    ROOT/'models/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt',OUT/'dataset.csv',OUT/'mic_regressor.joblib',
    ROOT/'phase1/artifacts/multiobjective/gflownet.pt',ROOT/'phase1/artifacts/final/library.fasta']
tracked.extend((OUT/'source/provenance_code').rglob('*.py'))
dump(OUT/'run_manifest.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in tracked})
text=f'''# Kết quả hoàn thiện Phase 1 và chạy Phase 2

## Phase 1: đã hoàn thành ensemble triển khai

- AMP-Diffusion checkpoint gốc với biến thể DDIM100 đã được chọn trước đó.
- ProtGPT2 tiền huấn luyện + LoRA rank16, train đủ 3 epoch.
- GFlowNet GRU được nâng cấp reward đa mục tiêu và train thêm {flow['steps']} bước
  Trajectory Balance: MIC dự đoán + novelty + lực đẩy khỏi tập anchor cố định + tính chất
  lý hóa, cùng LM prior. Các trọng số reward cố định, không chọn theo test/calibration.
- Thư viện cuối: đúng **50.000 chuỗi duy nhất**, quota 16.667 / 16.667 / 16.666,
  dài 8–50 aa, không trùng nguyên chuỗi với reference, dữ liệu fine-tune LM hoặc MIC.
- Kiểm tra tái sinh cùng seed và resume GFlowNet khớp {checks['gflownet_replay_sequences']}
  chuỗi đầu. Diffusion và ProtGPT2 đã có kiểm tra prefix riêng ở lượt trước.
- File: `phase1/artifacts/final/library.fasta` và `provenance.csv`.
- SHA-256: `{library['sha256']}`.

Đây là bản triển khai theo ba họ mô hình, không phải tái lập nguyên vẹn các bài báo.
ESM2-8M, GRU GFlowNet và ProtGPT2-LoRA là lựa chọn thực tế được ghi rõ.
Chưa chạy toàn bộ seqme Aggregation Score, chưa full replay 50.000 chuỗi trên máy khác,
chưa hoàn thiện đóng gói repo nộp bài/kiểm tra quyền phân phối dữ liệu.

## Phase 2: pipeline đã chạy xong; mục tiêu thống kê chưa đạt

Nguồn: DBAASP subset trong bản GRAMPA 2018, không phải tải mới toàn bộ DBAASP.
Sau lọc còn **{data['unique_sequences']} trình tự**, {data['similarity_components']} nhóm tương đồng.
Train/validation/calibration/test: {data['split_counts']}.
Đích dự đoán: median log10 MIC của mỗi trình tự trên các assay E. coli có sẵn;
không đại diện riêng strain hoặc điều kiện thử nghiệm của cuộc thi.

ESM2-8M embedding + 10 đặc trưng lý hóa; chọn **{cal['selected_model']}** bằng validation.

| Chỉ số test | Kết quả |
| --- | --- |
| MAE log10 MIC | {cal['test_mae_log10']:.4f} |
| MAE baseline median | {cal['test_median_baseline_mae']:.4f} |
| Spearman | {cal['test_spearman']:.4f} |
| Coverage hai phía, đặt mục tiêu 90% | **{100*cal['test_empirical_two_sided_coverage']:.2f}%** |
| Coverage cận trên một phía, đặt mục tiêu 90% | {100*cal['test_empirical_one_sided_coverage']:.2f}% |

Quantile hai phía: {cal['q_two_sided']:.4f} log10, tính đúng thứ hạng hữu hạn mẫu.
Giữ nguyên alpha=0,1 và MIC=16 µM; không điều chỉnh theo test hoặc số ứng viên đạt.
**Không được tuyên bố coverage 90% đã được xác thực hoặc hit-rate được bảo đảm.**
Có {data['diffusion_training_exact_overlap_by_split']['calibration']}/{cal['calibration_n']}
chuỗi calibration trùng dữ liệu train diffusion công bố. Còn khác biệt phân phối,
phụ thuộc theo nhóm, điều kiện assay và lựa chọn thích nghi của ứng viên sinh.
Ngay cả 25 phép thử độc lập với p=0,9, xác suất >=22 thành công chỉ là
{selection['counterexample_probability_at_least_22_of_25_if_independent_p90']:.4f}, không phải >=0,9.

## Xếp hạng thử nghiệm

- Đã chấm đủ {selection['library_count']} chuỗi: `phase2/artifacts/ranked_library.csv`.
- Qua novelty theo đúng normalized Indel/Levenshtein.ratio của cuộc thi:
  **{selection['reference_valid_count']}** chuỗi.
- Đồng thời upper bound thử nghiệm <=16 µM và similarity <=80%:
  **{selection['eligible_at_16uM']}** chuỗi.
- Trạng thái top100: `{selection['top100_status']}`.
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
'''
(ROOT/'phase2/KET_QUA.md').write_text(text,encoding='utf8')
(ROOT/'phase1/HOAN_THIEN.md').write_text('# Phase 1 hiện tại\n\nThư viện mới: `artifacts/final/library.fasta`.\n\nBáo cáo chung: [Phase 1 và Phase 2](../phase2/KET_QUA.md).\n\nCác báo cáo `KET_QUA_PHASE1.md` và thư viện cũ là lịch sử baseline, không phải bản mới nhất.\n',encoding='utf8')
print(json.dumps({'phase1':summary['phase1'],'phase2':summary['phase2'],'eligible':selection['eligible_at_16uM'],'coverage':cal['test_empirical_two_sided_coverage']},indent=2))
