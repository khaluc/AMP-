import math
import numpy as np
from core import conformal_quantile, properties, max_similarity
from rapidfuzz.distance import Indel

def test_finite_sample_rank_and_small_calibration():
    assert conformal_quantile(np.arange(1,10),.1)==9
    assert math.isinf(conformal_quantile([1,2,3],.1))
    assert conformal_quantile([4,1,3,2],.4)==3

def test_one_sided_signed_scores():
    assert conformal_quantile([-4,-3,-2,-1],.4)==-2

def test_similarity_and_descriptors():
    assert np.allclose(max_similarity(['AAAAAAAA','AAAAAAAC'],['AAAAAAAA']),[1,.875])
    p=properties(['AAAAAAAA','KKKKKKKK'])
    assert p.shape==(2,10) and np.isfinite(p).all()
    assert p[0,1]==0 and p[1,1]==1

def test_official_similarity_is_indel_not_edit_similarity():
    # One insertion and one deletion can make the official ratio higher.
    assert Indel.normalized_similarity('ACDEFGHI','CDEFGHIA')==.875
    assert np.allclose(max_similarity(['ACDEFGHI'],['CDEFGHIA']),[.75])
