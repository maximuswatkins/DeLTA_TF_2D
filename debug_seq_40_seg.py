#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage import io

from delta.data import predictGenerator_seg

# ---------- EDIT THESE ----------
MODEL = Path("/home/max/MaxWork/DeLTA_TF_New/models/My2D/seg/my2d_seg_model.keras")
SEQ = Path("/home/max/MaxWork/DeLTA_TF_New/data/My2D/CTC/test/05")
OUT = SEQ / "debug_predictions"
TARGET_SIZE = (256, 256)
# -------------------------------

OUT.mkdir(exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL, compile=False)

files = sorted(SEQ.glob("*.tif"))

print(f"Found {len(files)} frames")

gen = predictGenerator_seg(
    SEQ,
    files_list=files,
    target_size=TARGET_SIZE,
    crop_windows=False,
)

for i, fname in enumerate(files):

    x = next(gen)

    # Keras 3 generator format
    pred = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False).numpy()[0, :, :, 0]

    print(
        fname.name,
        "min =", pred.min(),
        "max =", pred.max(),
        "mean =", pred.mean(),
    )

    # Save raw probability map
    raw = (255 * pred).clip(0, 255).astype(np.uint8)
    io.imsave(OUT / f"{fname.stem}_raw.png", raw, check_contrast=False)

    # Save thresholded mask
    mask = (pred > 0.5).astype(np.uint8) * 255
    io.imsave(OUT / f"{fname.stem}_mask.png", mask, check_contrast=False)

print("Finished.")
