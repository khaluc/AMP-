from core import *
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

d=pd.read_csv(OUT/'source/grampa.csv')
counts={'raw_rows':len(d)}
d=d[(d.bacterium=='E. coli') & (d.database=='DBAASP') & d.datasource_has_modifications.eq(True) & d.is_modified.eq(False) & d.modifications.eq('[]') & d.unit.eq('uM')].copy()
counts['unmodified_ecoli_dbaasp_rows']=len(d)
# Do not uppercase: lowercase may encode stereochemistry in source data.
d=d[d.sequence.map(lambda s:isinstance(s,str) and valid(s)) & np.isfinite(d.value)]
d=d.drop_duplicates(['sequence','strain','value','url_source'])
d.to_csv(OUT/'source/eligible_measurements.csv',index=False)
rows=[]
for seq,g in d.groupby('sequence',sort=True):
    rows.append({'sequence':seq,'target_log10_uM':float(g.value.median()),'measurements':len(g),
        'strain_count':g.strain.nunique(), 'strains':'|'.join(sorted(g.strain.dropna().unique())),
        'log10_range':float(g.value.max()-g.value.min()),'urls':'|'.join(sorted(g.url_source.dropna().unique()))})
table=pd.DataFrame(rows); sequences=table.sequence.tolist(); n=len(table)
# Connected components: no pair with normalized edit similarity >0.8 crosses splits.
parent=list(range(n))
def find(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]]; i=parent[i]
    return i
for start in range(0,n,128):
    sim=process.cdist(sequences[start:start+128],sequences,scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float64)
    for i,j in zip(*np.where(sim>0.8)):
        a,b=find(start+int(i)),find(int(j))
        if a!=b:parent[b]=a
table['group']=[find(i) for i in range(n)]
indices=np.arange(n); groups=table.group.to_numpy()
def partition(idx,fraction,seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(idx,groups=groups[idx]))
    return idx[a],idx[b]
train_dev,held=partition(indices,0.30,42)
train,dev=partition(train_dev,1/7,43)
cal,test=partition(held,0.5,44)
table['split']=''
for label,idx in [('train',train),('validation',dev),('calibration',cal),('test',test)]:table.loc[idx,'split']=label
table.to_csv(OUT/'dataset.csv',index=False)
diff=set(fasta(ROOT/'starter-kits/ampdiffusion/data/training/training.fasta'))
new_train=set(fasta(ROOT/'phase1/artifacts/data/amp_train.fasta'))
new_train.update(fasta(ROOT/'phase1/artifacts/data/uniprot_train.fasta'))
for part in ['train','validation']:
    new_train.update(r['sequence'] for r in json.loads((ROOT/f'protein_lm/artifacts/data/{part}.json').read_text()))
counts.update({'unique_sequences':n,'similarity_components':table.group.nunique(),'largest_component':int(table.group.value_counts().max()),
    'split_counts':table.split.value_counts().to_dict(),
    'diffusion_training_exact_overlap_by_split':{s:int(g.sequence.isin(diff).sum()) for s,g in table.groupby('split')},
    'local_generator_data_exact_overlap_by_split':{s:int(g.sequence.isin(new_train).sum()) for s,g in table.groupby('split')},
    'endpoint':'Median log10(MIC/uM) per sequence across available E. coli assays/strains, DBAASP subset of 2018 GRAMPA snapshot',
    'label_transform':'Already log10 in source; verified src/load_data.py convert_result_to_rows. No second log transform.',
    'limitations':['Not strain/lab-specific; median endpoint is not an individual assay MIC.','Original censoring/assay protocol metadata unavailable; no recovery or imputation of censored values.','Homology-group splits are not IID sequence samples; report empirical coverage, not an unconditional exact guarantee.','Generator pretraining overlap and selection shift prevent top-100 hit-rate guarantees.'],
    'source_sha256':sha(OUT/'source/grampa.csv'),'dataset_sha256':sha(OUT/'dataset.csv')})
dump(OUT/'data_audit.json',counts); print(json.dumps(counts,indent=2))
