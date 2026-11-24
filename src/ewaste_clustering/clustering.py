"""Tahap 4 - Clustering K-Means++ dengan pemilihan K berdasarkan Silhouette Score.

Jumlah cluster tidak ditetapkan secara manual. K-Means++ dijalankan untuk setiap
K pada K_RANGE, lalu K dengan Silhouette Score tertinggi dipakai sebagai K
optimal.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import K_RANGE, N_INIT, SEED


def kmeans_pp(X, k, seed=SEED):
    """K-Means dengan inisialisasi K-Means++ (default scikit-learn) dan 10 kali inisialisasi."""
    return KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=seed).fit(X)


def select_k(X, k_range=K_RANGE):
    """Hitung Silhouette Score dan inersia untuk setiap K.

    Returns
    -------
    df_k   : DataFrame (K, silhouette, inertia)
    k_best : K dengan Silhouette Score tertinggi
    """
    rows = []
    for k in k_range:
        km = kmeans_pp(X, k)
        rows.append({"K": k, "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_})
    df_k = pd.DataFrame(rows)
    k_best = int(df_k.loc[df_k["silhouette"].idxmax(), "K"])
    return df_k, k_best


def fit_final(X, k_best):
    """Model akhir K-Means++ menggunakan K optimal hasil select_k."""
    return kmeans_pp(X, k_best)
