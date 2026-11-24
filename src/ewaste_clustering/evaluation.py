"""Tahap 6 - Evaluasi internal, stabilitas bootstrap, dan evaluasi eksternal."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
                             normalized_mutual_info_score, precision_recall_fscore_support,
                             silhouette_samples, silhouette_score)

from .clustering import kmeans_pp
from .config import N_BOOTSTRAP, SEED, TARGET_CLASSES


# ---------------------------------------------------------------- internal
def internal_metrics(X, cluster_labels):
    """Silhouette Score keseluruhan dan rata-rata silhouette tiap cluster."""
    s = silhouette_samples(X, cluster_labels)
    per_cluster = pd.DataFrame({"cluster": cluster_labels, "silhouette": s}) \
        .groupby("cluster")["silhouette"].agg(["count", "mean", "min"]).reset_index()
    return float(silhouette_score(X, cluster_labels)), per_cluster


# ---------------------------------------------------------------- stabilitas
def bootstrap_stability(X, ref_labels, k, n_boot=N_BOOTSTRAP, seed=SEED):
    """Uji stabilitas bootstrap.

    Pada setiap pengulangan, data diambil ulang dengan pengembalian, K-Means++
    dijalankan kembali dengan K yang sama, lalu hasilnya dibandingkan dengan
    cluster acuan pada citra unik yang terambil menggunakan ARI.
    """
    rng = np.random.RandomState(seed)
    n, rows = len(X), []
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        lab_b = kmeans_pp(X[idx], k, seed=seed + b + 1).labels_
        uniq, first = np.unique(idx, return_index=True)
        rows.append({"bootstrap": b + 1, "n_unique": len(uniq),
                     "ARI": adjusted_rand_score(ref_labels[uniq], lab_b[first])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- eksternal
def purity(y_true, cluster_labels):
    """Proporsi citra yang termasuk kategori mayoritas pada cluster-nya."""
    ct = pd.crosstab(cluster_labels, y_true)
    return float(ct.max(axis=1).sum() / ct.values.sum())


def external_metrics(y_true, cluster_labels, y_pred, classes=TARGET_CLASSES):
    """Metrik eksternal hasil clustering dan hasil pelabelan zero-shot."""
    clustering = {"purity": purity(y_true, cluster_labels),
                  "ARI": adjusted_rand_score(y_true, cluster_labels),
                  "NMI": normalized_mutual_info_score(y_true, cluster_labels)}
    labeling = {"accuracy": accuracy_score(y_true, y_pred),
                "F1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                "ARI": adjusted_rand_score(y_true, y_pred)}
    pr, rc, f1, sp = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame({"kelas": classes, "precision": pr, "recall": rc, "f1": f1, "support": sp})
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    return clustering, labeling, per_class, cm
