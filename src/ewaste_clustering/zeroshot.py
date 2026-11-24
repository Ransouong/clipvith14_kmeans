"""Tahap 5 - Penyusunan prompt dan pelabelan cluster dengan CLIP zero-shot.

Setiap kategori diubah menjadi kalimat "a photo of {kategori}", dikodekan
dengan text encoder CLIP, lalu dibandingkan dengan embedding citra. Skor
kemiripan dirata-ratakan per cluster dan label dengan skor tertinggi menjadi
nama cluster.
"""
import numpy as np
import pandas as pd
import torch

from .config import PROMPT_TEMPLATE, TARGET_CLASSES


def build_prompts(classes=TARGET_CLASSES, template=PROMPT_TEMPLATE):
    """Susun satu prompt berbahasa Inggris untuk setiap kategori."""
    return [template.format(c) for c in classes]


@torch.no_grad()
def encode_prompts(prompts, model, tokenizer, device):
    """Tokenisasi prompt, ekstraksi embedding teks, lalu normalisasi L2."""
    tokens = tokenizer(prompts).to(device)              # (n_prompt, 77)
    T = model.encode_text(tokens).float().cpu().numpy()  # (n_prompt, 1024)
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def label_clusters(E_l2, T_l2, cluster_labels, classes=TARGET_CLASSES):
    """Beri nama setiap cluster berdasarkan rata-rata cosine similarity tertinggi.

    Returns
    -------
    df_zs   : DataFrame skor rata-rata tiap cluster terhadap tiap prompt
    mapping : dict {cluster: label zero-shot}
    """
    S = E_l2 @ T_l2.T                                    # cosine similarity (n_citra, n_prompt)
    rows = []
    for c in np.unique(cluster_labels):
        mean_s = S[cluster_labels == c].mean(axis=0)
        best = int(mean_s.argmax())
        rows.append({"cluster": int(c), "n_images": int((cluster_labels == c).sum()),
                     **{cls: mean_s[j] for j, cls in enumerate(classes)},
                     "label_zeroshot": classes[best], "similarity_max": mean_s[best]})
    df_zs = pd.DataFrame(rows)
    mapping = dict(zip(df_zs["cluster"], df_zs["label_zeroshot"]))
    return df_zs, mapping
