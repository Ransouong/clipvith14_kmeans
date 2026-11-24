"""Tahap 2 - Ekstraksi fitur citra dengan image encoder CLIP ViT-H/14.

Setiap citra diubah ukurannya menjadi 224x224 (bicubic), dinormalisasi dengan
rata-rata dan simpangan baku data latih CLIP, lalu dilewatkan ke image encoder
sehingga menghasilkan satu vektor embedding berdimensi 1.024.
"""
import time

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile

from .config import BATCH_SIZE, CLIP_MODEL, CLIP_PRETRAINED

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_clip(device=None):
    """Muat model CLIP pretrained beserta fungsi pra-pemrosesan dan tokenizer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def extract_image_embeddings(paths, model, preprocess, device, batch_size=BATCH_SIZE):
    """Ekstraksi embedding citra per batch.

    Returns
    -------
    E : ndarray (n_citra, 1024) embedding mentah (belum dinormalisasi)
    log : list berisi waktu proses tiap batch
    """
    embeddings, log = [], []
    for b, start in enumerate(range(0, len(paths), batch_size)):
        t0 = time.time()
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]]
        x = torch.stack(batch).to(device)               # (B, 3, 224, 224)
        f = model.encode_image(x)                        # (B, 1024)
        embeddings.append(f.float().cpu().numpy())
        log.append({"batch": b + 1, "n_citra": len(batch), "detik": time.time() - t0})
    E = np.vstack(embeddings).astype(np.float32)
    return E, log
