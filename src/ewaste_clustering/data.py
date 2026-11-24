"""Tahap 1 - Input dan filterisasi data.

Kategori setiap citra hanya dibaca dari prefiks nama file. Label ini TIDAK
dipakai saat clustering, hanya sebagai pembanding (ground truth) saat evaluasi.
"""
import re
from pathlib import Path

import pandas as pd

from .config import TARGET_CLASSES, VALID_EXT


def get_class(filename):
    """Kembalikan kategori bila nama file diawali salah satu kategori, selain itu None.

    Pencocokan tidak membedakan huruf besar/kecil, dan garis bawah (_) serta
    tanda hubung (-) dianggap spasi, sehingga 'pcb_12.jpg' dan
    'Washing_Machine_3.jpg' tetap dikenali.
    """
    name = re.sub(r"[_\-]+", " ", filename).lower()
    for cls in TARGET_CLASSES:
        if name.startswith(cls.lower()):
            return cls
    return None


def load_dataset(data_dir):
    """Baca seluruh citra pada folder lalu pisahkan citra yang dipakai dan yang dibuang.

    Returns
    -------
    df_input : DataFrame (filename, label, path) citra yang dianalisis
    df_drop  : DataFrame (filename) citra tanpa keterangan kategori
    """
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VALID_EXT)
    df_all = pd.DataFrame({"filename": [p.name for p in paths], "path": paths})
    df_all["label"] = df_all["filename"].map(get_class)

    df_input = df_all[df_all["label"].notna()].reset_index(drop=True)
    df_drop = df_all[df_all["label"].isna()].reset_index(drop=True)
    return df_input, df_drop


def find_dataset_dir(root="/kaggle/input", folder="1_Electronic"):
    """Cari folder dataset di Kaggle (/kaggle/input/.../1_Electronic)."""
    return next(Path(root).rglob(folder))
