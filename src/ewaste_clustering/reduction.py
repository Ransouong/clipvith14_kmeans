"""Tahap 3 - Normalisasi L2 dan reduksi dimensi UMAP."""
import numpy as np
import umap

from .config import (SEED, UMAP_DIM_CLUSTER, UMAP_DIM_VIS, UMAP_METRIC,
                     UMAP_MIN_DIST_CLUSTER, UMAP_MIN_DIST_VIS, UMAP_NEIGHBORS)


def l2_normalize(E, eps=1e-12):
    """Bagi setiap vektor dengan norma L2-nya sehingga panjangnya menjadi 1."""
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / (norms + eps)).astype(np.float32)


def umap_reduce(E_l2, n_components, min_dist, seed=SEED):
    """Reduksi dimensi UMAP dengan 30 tetangga dan jarak kosinus."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=UMAP_NEIGHBORS,
                        min_dist=min_dist, metric=UMAP_METRIC,
                        random_state=seed, low_memory=False)
    return reducer.fit_transform(E_l2), reducer


def reduce_for_clustering(E_l2):
    """UMAP 50 dimensi (min_dist = 0,0) sebagai ruang clustering."""
    return umap_reduce(E_l2, UMAP_DIM_CLUSTER, UMAP_MIN_DIST_CLUSTER)[0]


def reduce_for_visualization(E_l2):
    """UMAP 2 dimensi (min_dist = 0,1) khusus untuk visualisasi."""
    return umap_reduce(E_l2, UMAP_DIM_VIS, UMAP_MIN_DIST_VIS)[0]
