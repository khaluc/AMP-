"""Independent audits of split isolation, saved model, assembly and generated prefix."""
from core import *
import pandas as pd, joblib
from types import SimpleNamespace
from gflownet import generate, DEST

data=pd.read_csv(OUT/'dataset.csv')
assert data.groupby('group').split.nunique().max()==1
assert data.sequence.is_unique and data.sequence.map(valid).all()
held=pd.read_csv(OUT/'heldout_predictions.csv')
info=json.loads((OUT/'calibration.json').read_text())
cal=held[held.split=='calibration'];test=held[held.split=='test']
q=conformal_quantile(abs(cal.target_log10_uM-cal.prediction_log10_uM))
assert np.isclose(q,info['q_two_sided'])
coverage=float((abs(test.target_log10_uM-test.prediction_log10_uM)<=q).mean())
assert np.isclose(coverage,info['test_empirical_two_sided_coverage'])
assert info['model_sha256']==sha(OUT/'mic_regressor.joblib')
model=joblib.load(OUT/'mic_regressor.joblib'); x=np.load(OUT/'dataset_features.npy')
assert np.allclose(model.predict(x),held.prediction_log10_uM,atol=1e-8)
library=fasta(ROOT/'phase1/artifacts/final/library.fasta')
assert len(library)==len(set(library))==50000 and all(valid(s) for s in library)
ranked=pd.read_csv(OUT/'ranked_library.csv');assert set(library)==set(ranked.sequence)
assert not set(library)&set(fasta(ROOT/'data/reference/antibacterial.fasta'))
assert not set(library)&set(data.sequence)
eligible=pd.read_csv(OUT/'eligible_candidates.csv')
assert (eligible.upper_uM<=16).all() and (eligible.max_reference_similarity<=.8).all()
replay=DEST/'replay.fasta'
if replay.exists():raise FileExistsError('Replay path exists; preserve previous evidence')
args=SimpleNamespace(output=str(replay),resume=False,count=1)
generate(args);first=fasta(replay)
assert first==fasta(DEST/'pool.fasta')[:len(first)]
args.resume=True;args.count=len(first)+1;generate(args);second=fasta(replay)
assert second==fasta(DEST/'pool.fasta')[:len(second)]
from rapidfuzz.distance import Indel
subset=ranked.sort_values('id').head(10)
refs=fasta(ROOT/'data/reference/antibacterial.fasta')
for row in subset.itertuples():
    exact=max(Indel.normalized_similarity(row.sequence,r) for r in refs)
    assert np.isclose(exact,row.max_reference_similarity,atol=1e-12)
dump(OUT/'verification.json',{'split_group_isolation':True,'dataset_unique':True,'finite_sample_quantile_recomputed':True,
    'model_reload_predictions_match':True,'library_structural_and_exact_overlap_checks':True,
    'gflownet_same_seed_and_resume_prefix':True,'gflownet_replay_sequences':len(second),
    'reference_similarity_scalar_crosscheck':10,'actual_test_coverage':coverage,
    'statistical_target_met':coverage>=.9,'scope':'Reproducibility checks are prefixes on this hardware, not a complete 50000 replay.'})
print('Independent technical audit passed; empirical coverage target met:',coverage>=.9)
