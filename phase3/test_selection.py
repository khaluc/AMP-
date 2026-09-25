import numpy as np
from itertools import combinations
from selection import greedy,objective

def test_diminishing_returns_and_monotonicity():
    s=np.array([[1,.8,.1,0],[0,.1,.7,1.]])
    q=np.array([.7,.8,.9,1.]);k=3
    a=[0];b=[0,1];j=2
    da=objective(s,q,a+[j],k)-objective(s,q,a,k)
    db=objective(s,q,b+[j],k)-objective(s,q,b,k)
    assert da>=db>=0

def test_greedy_against_exact_small_problem():
    rng=np.random.default_rng(7)
    s=rng.random((6,8));q=rng.random(8);k=3
    chosen,trace=greedy(s,q,k)
    optimum=max(objective(s,q,list(ids),k) for ids in combinations(range(8),k))
    assert objective(s,q,chosen,k)>=(1-(1-1/k)**k)*optimum-1e-12
    assert len(set(chosen))==k
    for i,row in enumerate(trace):
        expected=objective(s,q,chosen[:i+1],k)-objective(s,q,chosen[:i],k)
        assert np.isclose(row['marginal_gain'],expected)

def test_exact_ties_stable_and_redundancy_covered():
    s=np.array([[1,1,0],[0,0,1.]])
    chosen,_=greedy(s,np.ones(3),2)
    assert chosen==[0,2]
