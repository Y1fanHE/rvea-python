import numpy as np

def adp_selection(X, F, ref_dirs, theta0, ref_angle):
    n_refdirs, _ = ref_dirs.shape

    z_min = np.min(F, axis=0)
    F_shift = F - z_min
    F_norm = F_shift / ( np.linalg.norm(F_shift, axis=1).reshape(-1, 1) + 1e-14 )

    cosine = np.dot(F_norm, ref_dirs.T)
    arccosine = np.arccos( np.clip(cosine, 0, 1) )
    max_idx_cosine = np.argmax(cosine, axis=1)
    subpops = dict( zip( np.arange(n_refdirs), [[]] * n_refdirs) )
    max_idx_cosine_unique = set(max_idx_cosine)
    for idx in max_idx_cosine_unique:
        tmp = list(np.where(max_idx_cosine == idx)[0])
        subpops.update({idx: tmp})

    selection = []
    for i in range(n_refdirs):
        if len(subpops[i]) != 0:
            subpop = subpops[i]
            F_sub = F[subpop]

            # adp calculation
            arccosine_sub = arccosine[subpop, i]
            arccosine_sub = arccosine_sub / ref_angle[i]
            d1 = np.linalg.norm(F_sub, axis=1)
            d = d1 * (1 + theta0 * arccosine_sub)

            min_idx_adp = np.argmin(d)
            selection.append( subpop[min_idx_adp] )

    return X[selection, :], F[selection]