from pathlib import Path
import sys, os, json, math, time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase2'))
from core import fasta,valid,sha,dump,properties
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits
from rapidfuzz import process
from rapidfuzz.distance import Indel
from selection import greedy,objective

OUT=ROOT/'phase3/artifacts';OUT.mkdir(parents=True,exist_ok=True)
P2=ROOT/'phase2/artifacts';LIB=ROOT/'phase1/artifacts/final/library.fasta'

def describe(frame,indices,sim,quality,clusters,z):
    ids=frame.id.tolist(); seq=frame.sequence.tolist()
    mat=process.cdist(seq,seq,scorer=Indel.normalized_similarity,dtype=np.float64,workers=4)
    upper=mat[np.triu_indices(len(seq),1)]
    rng=np.random.default_rng(3042); draws=[]
    for _ in range(2000):
        rows=rng.choice(len(frame),25,replace=False)
        draws.append(len(np.unique(frame.cluster.to_numpy()[rows])))
    return {'n':len(frame),'generators':frame.generator.value_counts().to_dict(),'hard_clusters':int(frame.cluster.nunique()),
        'facility_coverage':float(sim[:,indices].max(axis=1).mean()),'objective':objective(sim,quality,indices,100),
        'mean_pairwise_indel_distance':float((1-upper).mean()),'pairs_above_80pct_similarity':int((upper>.8).sum()),
        'median_upper_uM':float(frame.upper_uM.median()),'max_upper_uM':float(frame.upper_uM.max()),
        'max_reference_similarity':float(frame.max_reference_similarity.max()),
        'random25_distinct_clusters_mean':float(np.mean(draws)),
        'random25_distinct_clusters_p05_p50_p95':np.quantile(draws,[.05,.5,.95]).tolist()}

def main():
    begun=time.time()
    ranked=pd.read_csv(P2/'ranked_library.csv').sort_values('id').reset_index(drop=True)
    sequences=fasta(LIB)
    assert len(ranked)==50000 and ranked.sequence.tolist()==sequences
    cache=P2/'library_features';manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest=={'library':sha(LIB),'core':sha(ROOT/'phase2/core.py'),'model':sha(P2/'mic_regressor.joblib')}
    arrays=[];hashes={}
    for start in range(0,50000,512):
        path=cache/f'{start:06d}.npz';saved=np.load(path);x=saved['features']
        assert x.shape==(min(512,50000-start),330) and np.isfinite(x).all()
        assert np.allclose(x[:,320:],properties(sequences[start:start+512]),atol=1e-5)
        arrays.append(x);hashes[path.name]=sha(path)
    x=np.concatenate(arrays)
    # Equal total weight for standardized embedding and descriptor blocks.
    emb=StandardScaler().fit(x[:,:320]);desc=StandardScaler().fit(x[:,320:])
    z=np.concatenate([emb.transform(x[:,:320])*math.sqrt(.5/320),desc.transform(x[:,320:])*math.sqrt(.5/10)],axis=1)
    print('Feature cache validated; clustering 50000 sequences.',flush=True)
    km=MiniBatchKMeans(n_clusters=100,batch_size=2048,n_init=10,max_iter=200,random_state=42,reassignment_ratio=.01)
    with threadpool_limits(limits=4):
        clusters=km.fit_predict(z)
    ranked['cluster']=clusters
    assert len(np.unique(clusters))==100
    ranked[['id','sequence','generator','cluster']].to_csv(OUT/'library_clusters.csv',index=False)
    joblib.dump({'embedding_scaler':emb,'descriptor_scaler':desc,'clustering':km},OUT/'clustering.joblib')
    eligible=ranked[(ranked.upper_uM<=16)&(ranked.max_reference_similarity<=.8)].copy()
    assert len(eligible)==167
    positions=eligible.index.to_numpy();eligible=eligible.reset_index(drop=True)
    d2=pairwise_distances(km.cluster_centers_,z[positions],metric='sqeuclidean')
    cc=pairwise_distances(km.cluster_centers_)
    np.fill_diagonal(cc,np.inf);bandwidth=float(np.median(cc.min(axis=1)))
    assert bandwidth>0
    sim=np.exp(-d2/(2*bandwidth**2))
    quality=eligible.upper_uM.min()/eligible.upper_uM.to_numpy()
    chosen,trace=greedy(sim,quality,100,.8)
    again,_=greedy(sim,quality,100,.8);assert chosen==again
    selected=eligible.iloc[chosen].copy();selected.insert(0,'portfolio_rank',np.arange(1,101))
    selected['selection_gain']=[t['marginal_gain'] for t in trace]
    selected['mechanism_status']='unassigned_no_mechanistic_validation'
    selected.to_csv(OUT/'top100_portfolio_exploratory.csv',index=False)
    with (OUT/'top100_portfolio_exploratory.fasta').open('w',encoding='ascii',newline='\n') as f:
        for row in selected.itertuples():f.write(f'>{row.id}\n{row.sequence}\n')
    eligible.to_csv(OUT/'eligible_with_clusters.csv',index=False)
    np.savez(OUT/'selection_inputs.npz',similarity=sim,quality=quality,chosen=chosen,centroids=km.cluster_centers_)
    dump(OUT/'greedy_trace.json',trace)
    base=pd.read_csv(P2/'top100_exploratory.csv')
    lookup={s:i for i,s in enumerate(eligible.sequence)}
    baseline_indices=[lookup[s] for s in base.sequence]
    baseline=eligible.iloc[baseline_indices]
    a=describe(baseline,baseline_indices,sim,quality,clusters,z)
    b=describe(selected,chosen,sim,quality,clusters,z)
    cluster_rows=[]
    for group,frame in ranked.groupby('cluster'):
        e=eligible[eligible.cluster==group];s=selected[selected.cluster==group]
        cluster_rows.append({'cluster':int(group),'library_count':len(frame),'eligible_count':len(e),'selected_count':len(s),'mechanism_status':'unknown'})
    pd.DataFrame(cluster_rows).to_csv(OUT/'cluster_summary.csv',index=False)
    # Representative IDs are an index for later research, not mechanistic assignments.
    representatives=[]
    for c in sorted(selected.cluster.unique()):
        f=selected[selected.cluster==c]
        pos=[sequences.index(s) for s in f.sequence]
        i=int(np.argmin(np.sum((z[pos]-km.cluster_centers_[c])**2,axis=1)))
        representatives.append(f.iloc[i])
    pd.DataFrame(representatives).to_csv(OUT/'representatives_for_future_validation.csv',index=False)
    checks={'unique_100':len(selected)==selected.sequence.nunique()==100,'subset_library':set(selected.sequence)<=set(sequences),
      'thresholds_preserved':bool((selected.upper_uM<=16).all() and (selected.max_reference_similarity<=.8).all()),
      'same_inputs_selection_replay':chosen==again,'sequence_valid':all(valid(s) for s in selected.sequence),
      'scalar_reference_similarity':True}
    refs=fasta(ROOT/'data/reference/antibacterial.fasta')
    for row in selected.itertuples():
        maximum=max(Indel.normalized_similarity(row.sequence,r) for r in refs)
        assert maximum<=.8 and np.isclose(maximum,row.max_reference_similarity,atol=1e-12)
    assert all(checks.values())
    result={'config':{'seed':42,'clusters':100,'eligible_upper_uM':16,'reference_similarity_max':.8,'facility_weight':.8,'quality_weight':.2,'rbf_bandwidth':bandwidth,'quota_per_generator':None},
      'eligible_count':len(eligible),'eligible_hard_clusters':int(eligible.cluster.nunique()),
      'empty_of_eligible_clusters':100-int(eligible.cluster.nunique()),'baseline':a,'portfolio':b,
      'shared_with_baseline':len(set(base.sequence)&set(selected.sequence)),'checks':checks,
      'cluster_representatives':len(representatives),'seconds':time.time()-begun,
      'limitations':['Phase2 empirical interval coverage is 77.1%, not validated at 90%.','Clusters are geometric groups, not mechanisms of action.','Fixed threshold may leave many clusters without an eligible candidate.','Facility-location guarantee concerns the specified surrogate objective only, not potency, MoA or hit-rate.','Random25 simulation measures geometric cluster coverage only.','Optional MARTINI simulations not performed; no structures/topologies or mechanistic evidence available.']}
    dump(OUT/'result.json',result)
    dump(OUT/'manifest.json',{'library_sha256':sha(LIB),'ranked_sha256':sha(P2/'ranked_library.csv'),'feature_chunks':hashes,
        'scripts':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'environment_sha256':sha(ROOT/'phase1/uv.lock'),
        'top100_sha256':sha(OUT/'top100_portfolio_exploratory.fasta')})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
