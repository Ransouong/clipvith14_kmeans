"""Menjalankan seluruh alur penelitian dari citra mentah sampai evaluasi.

Contoh:
    python run_pipeline.py --data-dir /kaggle/input/bdc2026-dataset/train/1_Electronic
    python run_pipeline.py                       # di Kaggle, folder dicari otomatis

Semua keluaran (embedding, hasil clustering, label, dan metrik) disimpan di --out-dir.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ewaste_clustering import clustering, data, evaluation, features, reduction, zeroshot  # noqa: E402
from ewaste_clustering.config import SEED, TARGET_CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder 1_Electronic; bila kosong dicari di /kaggle/input")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Input dan filterisasi data
    data_dir = Path(args.data_dir) if args.data_dir else data.find_dataset_dir()
    df_input, df_drop = data.load_dataset(data_dir)
    y_true = df_input["label"].to_numpy()
    print(f"[1] citra digunakan: {len(df_input)} | tidak digunakan: {len(df_drop)}")
    df_input[["filename", "label"]].to_csv(out / "ground_truth_labels.csv", index=False)

    # 2. Ekstraksi fitur CLIP ViT-H/14
    model, preprocess, tokenizer, device = features.load_clip()
    E, _ = features.extract_image_embeddings(list(df_input["path"]), model, preprocess, device)
    np.save(out / "emb_raw.npy", E)
    print(f"[2] embedding citra: {E.shape} ({device})")

    # 3. Normalisasi L2 dan reduksi dimensi UMAP
    E_l2 = reduction.l2_normalize(E)
    X50 = reduction.reduce_for_clustering(E_l2)
    X2 = reduction.reduce_for_visualization(E_l2)
    np.save(out / "umap_50d.npy", X50); np.save(out / "umap_2d.npy", X2)
    print(f"[3] UMAP: {X50.shape} untuk clustering, {X2.shape} untuk visualisasi")

    # 4. K-Means++ dengan K dari Silhouette Score tertinggi
    df_k, k_best = clustering.select_k(X50)
    df_k.to_csv(out / "kmeans_k_selection.csv", index=False)
    km = clustering.fit_final(X50, k_best)
    labels = km.labels_
    print(f"[4] K optimal = {k_best} (silhouette {df_k['silhouette'].max():.4f})")

    # 5. Prompt dan pelabelan zero-shot
    prompts = zeroshot.build_prompts()
    T_l2 = zeroshot.encode_prompts(prompts, model, tokenizer, device)
    df_zs, mapping = zeroshot.label_clusters(E_l2, T_l2, labels)
    df_zs.to_csv(out / "zs_cluster_labels.csv", index=False)
    y_pred = np.array([mapping[c] for c in labels])
    print(f"[5] label cluster: {mapping}")

    # 6. Evaluasi internal, bootstrap, dan eksternal
    sil, sil_cluster = evaluation.internal_metrics(X50, labels)
    sil_cluster.to_csv(out / "silhouette_per_cluster.csv", index=False)
    df_boot = evaluation.bootstrap_stability(X50, labels, k_best)
    df_boot.to_csv(out / "bootstrap_ari.csv", index=False)
    m_clu, m_lab, per_class, cm = evaluation.external_metrics(y_true, labels, y_pred)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)
    cm.to_csv(out / "confusion_matrix.csv")

    df_input.assign(cluster=labels, label_zeroshot=y_pred, umap_x=X2[:, 0], umap_y=X2[:, 1]) \
        .drop(columns="path").to_csv(out / "kmeans_cluster_results.csv", index=False)

    summary = {
        "n_input": len(df_input), "n_dropped": len(df_drop),
        "K": k_best, "silhouette": sil, "inertia": float(km.inertia_),
        "bootstrap_ari_mean": float(df_boot["ARI"].mean()), "bootstrap_ari_std": float(df_boot["ARI"].std(ddof=0)),
        "bootstrap_ari_ci95": [float(v) for v in np.percentile(df_boot["ARI"], [2.5, 97.5])],
        "clustering": m_clu, "zeroshot": m_lab,
        "n_unique_labels": len(set(mapping.values())), "classes": TARGET_CLASSES,
        "runtime_seconds": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
