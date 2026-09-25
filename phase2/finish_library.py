from core import *
import csv, pandas as pd
dest=ROOT/'phase1/artifacts/final'
known=set(fasta(ROOT/'data/reference/antibacterial.fasta'))
known.update(pd.read_csv(OUT/'dataset.csv').sequence)
for part in ['train','validation']:
    known.update(r['sequence'] for r in json.loads((ROOT/f'protein_lm/artifacts/data/{part}.json').read_text()))
sources=[('diffusion_ddim100',ROOT/'phase1/artifacts/pools/diffusion_ddim100.fasta',16667),
 ('protgpt2_lora',ROOT/'protein_lm/artifacts/pool/protgpt2_lora.fasta',16667),
 ('gflownet_multiobjective',ROOT/'phase1/artifacts/multiobjective/pool.fasta',16666)]
seen=set();rows=[];audit={}
for name,path,count in sources:
    kept=[]
    for s in fasta(path):
        if valid(s) and s not in known and s not in seen:kept.append(s);seen.add(s)
        if len(kept)==count:break
    if len(kept)!=count:raise RuntimeError(f'{name}: insufficient candidates')
    rows.extend((name,s) for s in kept);audit[name]={'selected':len(kept),'pool_sha256':sha(path)}
write_fasta(dest/'library.fasta',[s for _,s in rows])
with (dest/'provenance.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.writer(f);w.writerow(['id','generator','sequence'])
    w.writerows((f'seq_{i:06d}',name,s) for i,(name,s) in enumerate(rows,1))
assert len(rows)==len(seen)==50000 and not seen&known
dump(dest/'audit.json',{'count':50000,'unique':50000,'exact_known_overlap':0,'sources':audit,'sha256':sha(dest/'library.fasta'),'scope':'Three-family operational ensemble; no potency or exact GFlowNet convergence guarantee.'})
print('Final 50000 library assembled',sha(dest/'library.fasta'))
