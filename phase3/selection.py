"""Monotone submodular facility-location + nonnegative modular quality."""
import numpy as np

def objective(similarity, quality, indices, k, weight=.8):
    if not len(indices):return 0.
    return float(weight*similarity[:,indices].max(axis=1).mean()+(1-weight)*quality[indices].sum()/k)

def greedy(similarity,quality,k,weight=.8):
    similarity=np.asarray(similarity,dtype=float);quality=np.asarray(quality,dtype=float)
    if similarity.ndim!=2 or similarity.shape[1]!=len(quality) or not 0<k<=len(quality) or not 0<=weight<=1:
        raise ValueError('Invalid dimensions/count/weight')
    if not np.isfinite(similarity).all() or not np.isfinite(quality).all() or np.any(similarity<0) or np.any(quality<0):
        raise ValueError('Finite nonnegative similarities and quality required')
    cover=np.zeros(similarity.shape[0]);chosen=[];trace=[];available=np.ones(len(quality),bool)
    for step in range(k):
        gains=weight*np.maximum(similarity-cover[:,None],0).mean(axis=0)+(1-weight)*quality/k
        gains[~available]=-np.inf
        j=int(np.argmax(gains)) # Stable candidate ID order breaks exact ties.
        chosen.append(j);available[j]=False;cover=np.maximum(cover,similarity[:,j])
        trace.append({'step':step+1,'candidate_index':j,'marginal_gain':float(gains[j]),'objective':objective(similarity,quality,chosen,k,weight)})
    return chosen,trace
