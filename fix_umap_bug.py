import numba
import pynndescent
import numpy as np


def fix_umap_bug():
    """Fixing umap bug. https://github.com/lmcinnes/pynndescent/issues/163"""

    @numba.njit(fastmath=True)
    def correct_alternative_cosine(ds):
        result = np.empty_like(ds)
        for i in range(ds.shape[0]):
            result[i] = 1.0 - np.power(2.0, ds[i])
        return result

    pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
    pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
    pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine
