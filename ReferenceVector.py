import numpy as np
from scipy.spatial.distance import cdist

# reference vector generation
def das_dennis(n_part, n_obj):
    if n_part == 0:
        return np.full((1, n_obj), 1 / n_obj)
    else:
        ref_dirs = []
        ref_dir = np.full(n_obj, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_part, n_part, 0)
        return np.concatenate(ref_dirs, axis=0)

def das_dennis_recursion(ref_dirs, ref_dir, n_part, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_part)
        ref_dir = ref_dir / np.sqrt( np.sum(ref_dir ** 2) )
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_part)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_part, beta - i, depth + 1)

def neighboring_angle(ref_dirs):
    cosine_refdirs = np.dot(ref_dirs, ref_dirs.T)
    sorted_cosine_refdirs = - np.sort(- cosine_refdirs, axis=1)
    arccosine_refdirs = np.arccos( np.clip(sorted_cosine_refdirs[:,1], 0, 1) )
    return arccosine_refdirs