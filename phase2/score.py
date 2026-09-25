from core import *
import pandas as pd, joblib, time
from rapidfuzz.distance import Indel
from scipy.stats import binom

library=ROOT/'phase1/artifacts/final/library.fasta'
sequences=fasta(library); metadata=json.loads((OUT/'calibration.json').read_text())
model=joblib.load(OUT/'mic_regressor.joblib'); extractor=Features()
cache=OUT/'library_features'; cache.mkdir(exist_ok=True)
fingerprint={'library':sha(library),'core':sha(Path(__file__).with_name('core.py')),'model':sha(OUT/'mic_regressor.joblib')}
manifest=cache/'manifest.json'
if manifest.exists() and json.loads(manifest.read_text())!=fingerprint:raise RuntimeError('Feature cache mismatch')
dump(manifest,fingerprint)
refs=sorted(set(fasta(ROOT/'data/reference/antibacterial.fasta')))
parts=[]; begun=time.time()
for offset in range(0,len(sequences),512):
    path=cache/f'{offset:06d}.npz'; seq=sequences[offset:offset+512]
    if path.exists():
        saved=np.load(path); x=saved['features']; sim=saved['similarity']
    else:
        x=extractor(seq)
        # Organizer uses python-Levenshtein.ratio = normalized Indel similarity,
        # NOT normalized Levenshtein edit distance. Use float64 at the 0.8 boundary.
        sim=[]
        for i in range(0,len(seq),128):
            scores=process.cdist(seq[i:i+128],refs,scorer=Indel.normalized_similarity,workers=4,dtype=np.float64)
            sim.extend(scores.max(axis=1))
        sim=np.array(sim)
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez(f,features=x,similarity=sim)
        temp.replace(path)
    pred=model.predict(x)
    parts.append(pd.DataFrame({'sequence':seq,'prediction_log10_uM':pred,'lower_log10_uM':pred-metadata['q_two_sided'],
        'upper_log10_uM':pred+metadata['q_two_sided'],'one_sided_upper_log10_uM':pred+metadata['q_one_sided'],'max_reference_similarity':sim}))
    if offset%5120==0:print(json.dumps({'processed':min(offset+512,len(sequences)),'seconds':time.time()-begun}),flush=True)
table=pd.concat(parts,ignore_index=True)
table.insert(0,'id',[f'seq_{i:06d}' for i in range(1,len(table)+1)])
source=pd.read_csv(ROOT/'phase1/artifacts/final/provenance.csv')
assert table.sequence.tolist()==source.sequence.tolist()
table['generator']=source.generator
for name in ['prediction','lower','upper','one_sided_upper']:table[name+'_uM']=10**table[name+'_log10_uM']
table['passes_reference_80pct']=table.max_reference_similarity<=0.8
table['passes_exploratory_upper_16uM']=table.upper_uM<=16
table=table.sort_values(['upper_log10_uM','id'],kind='stable')
table.to_csv(OUT/'ranked_library.csv',index=False)
eligible=table[table.passes_reference_80pct & table.passes_exploratory_upper_16uM]
eligible.to_csv(OUT/'eligible_candidates.csv',index=False)
write_fasta(OUT/'eligible_candidates.fasta',eligible.sequence.tolist())
if len(eligible)>=100:
    eligible.head(100).to_csv(OUT/'top100_exploratory.csv',index=False)
    write_fasta(OUT/'top100_exploratory.fasta',eligible.head(100).sequence.tolist())
else:
    # Explicitly named diagnostic, never substitute for a threshold-qualified top100.
    table[table.passes_reference_80pct].head(100).to_csv(OUT/'diagnostic_best100_not_threshold_qualified.csv',index=False)
summary={'library_sha256':sha(library),'model_sha256':sha(OUT/'mic_regressor.joblib'),'library_count':len(table),
    'reference_valid_count':int(table.passes_reference_80pct.sum()),'eligible_at_16uM':len(eligible),
    'top100_status':'exploratory_candidates_written' if len(eligible)>=100 else 'insufficient_candidates_no_threshold_qualified_top100',
    'sensitivity_counts_not_retuned':{str(t):int(((table.upper_uM<=t)&table.passes_reference_80pct).sum()) for t in [8,16,32,64]},
    'by_generator':table.groupby('generator').agg(count=('sequence','size'),median_pred_log10=('prediction_log10_uM','median'),median_similarity=('max_reference_similarity','median')).to_dict('index'),
    'counterexample_probability_at_least_22_of_25_if_independent_p90':float(binom.sf(21,25,.9)),
    'claim':'Exploratory historical-endpoint bounds only. Not calibrated for adaptively generated, selected candidates. No 22/25 guarantee.',
    'peak_reserved_mib':extractor.torch.cuda.max_memory_reserved()/2**20}
dump(OUT/'selection_audit.json',summary);print(json.dumps(summary,indent=2))
