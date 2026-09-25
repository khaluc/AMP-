from pathlib import Path
import os, sys, math, json
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'artifacts'
sys.path.insert(0, str(ROOT / 'phase1'))
from common import AA, fasta, valid, write_fasta, dump, sha
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
def properties(sequences):
    rows=[]
    for s in sequences:
        n=len(s); h=np.array([KD[a] for a in s]); angle=np.arange(n)*math.pi*100/180
        rows.append([n, (s.count('K')+s.count('R')-s.count('D')-s.count('E'))/n,
            h.mean(), abs(np.sum(h*np.exp(1j*angle)))/n,
            sum(a in 'AILMFWVY' for a in s)/n, sum(a in 'FWY' for a in s)/n,
            s.count('C')/n, s.count('P')/n, s.count('G')/n,
            sum(a==b for a,b in zip(s,s[1:]))/max(1,n-1)])
    return np.array(rows,dtype=np.float32)

def max_similarity(sequences, references, block=128):
    result=[]
    for i in range(0,len(sequences),block):
        scores=process.cdist(sequences[i:i+block],references,scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32)
        result.extend(scores.max(axis=1).tolist())
    return np.array(result)

def conformal_quantile(scores, alpha=0.1):
    scores=np.asarray(scores,dtype=float)
    if not 0<alpha<1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite scores, n>0 and 0<alpha<1 required')
    k=math.ceil((len(scores)+1)*(1-alpha))
    return float('inf') if k>len(scores) else float(np.sort(scores)[k-1])

class Features:
    def __init__(self):
        import torch, esm
        from networks import seed_all
        seed_all(42)
        self.torch=torch
        self.model,self.alphabet=esm.pretrained.esm2_t6_8M_UR50D()
        self.model=self.model.cuda().eval().requires_grad_(False)
        self.convert=self.alphabet.get_batch_converter()
    def __call__(self,sequences,bs=128):
        values=[]
        with self.torch.no_grad():
            for offset in range(0,len(sequences),bs):
                batch=sequences[offset:offset+bs]
                _,_,tokens=self.convert([(str(i),s) for i,s in enumerate(batch)])
                reps=self.model(tokens.cuda(),repr_layers=[6],return_contacts=False)['representations'][6]
                values.extend([reps[i,1:len(s)+1].mean(0).cpu().numpy() for i,s in enumerate(batch)])
        return np.concatenate([np.array(values), properties(sequences)],axis=1)
