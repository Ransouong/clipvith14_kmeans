"""Konfigurasi penelitian.

Seluruh nilai di sini sama dengan yang dipakai pada naskah skripsi, sehingga
menjalankan ulang pipeline akan menghasilkan angka yang sama.
"""

SEED = 42

# Sepuluh kategori sampah elektronik (diambil dari prefiks nama file)
TARGET_CLASSES = [
    "Battery", "Keyboard", "Microwave", "Mobile", "Mouse",
    "PCB", "Player", "Printer", "Television", "Washing Machine",
]
VALID_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# Model CLIP (pustaka open_clip)
CLIP_MODEL = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
BATCH_SIZE = 64

# UMAP
UMAP_NEIGHBORS = 30
UMAP_METRIC = "cosine"
UMAP_DIM_CLUSTER = 50      # ruang untuk clustering
UMAP_MIN_DIST_CLUSTER = 0.0
UMAP_DIM_VIS = 2           # ruang untuk visualisasi
UMAP_MIN_DIST_VIS = 0.1

# K-Means++
K_RANGE = range(2, 26)     # K yang diuji: 2 sampai 25
N_INIT = 10

# Bootstrap
N_BOOTSTRAP = 50

# Zero-shot
PROMPT_TEMPLATE = "a photo of {}"
