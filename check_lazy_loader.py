from pprint import pprint
from pathlib import Path
import numpy as np
import delta

# Replace these imports with your two loaders
from delta.data import load_training_dataset_seg as load_old
from delta.data_lazy import load_training_dataset_seg as load_lazy

dataset_path = Path("/home/max/MaxWork/DeLTA_TF_New/data/My2D/CTC/")
config = delta.config.DEFAULT_CONFIG_2D

old_ds = load_old(
    dataset_path=dataset_path / "val",
    target_size=config.target_size_seg,
    crop=config.presets=='2D',
    kw_data_aug={},
    stack=True,
    mode="evaluation",
)

lazy_ds = load_lazy(
    dataset_path=dataset_path / "val",
    target_size=config.target_size_seg,
    crop=config.presets=='2D',
    kw_data_aug={},
    stack=True,
    mode="evaluation",
)

# If these are Sequence-like objects:
x_old, y_old = old_ds[0]
x_new, y_new = lazy_ds[0]

print("OLD")
print("image:", x_old.shape, x_old.min(), x_old.max())
print("mask: ", y_old.shape, y_old.min(), y_old.max())

print("\nLAZY")
print("image:", x_new.shape, x_new.min(), x_new.max())
print("mask: ", y_new.shape, y_new.min(), y_new.max())

print("\nCOMPARISON")
print("image equal:", np.allclose(x_old, x_new))
print("mask equal: ", np.allclose(y_old, y_new))