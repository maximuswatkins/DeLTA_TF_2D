"""
Run the DeLTA pipeline on a demo movie.

In order to use on bioformats-compatible files such .nd2, .czi, .oib, ome-tiff
 etc files please install python-bioformats first.

@author: jeanbaptiste
"""
import re
import delta
from pathlib import Path
import time

import csv #for saving inference time to csv
import argparse #for specifying split (train, val, test) when running the pipeline

# dataset = 'DynamicNuclearNet-tracking-v1_0' # ['moma',''DynamicNuclearNet-tracking-v1_0']
# dataset = 'moma' # ['moma',''DynamicNuclearNet-tracking-v1_0']
dataset = 'My2D'

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val", "test"], default="test")
args = parser.parse_args()

datapath = Path('/home/max/MaxWork/DeLTA_TF_New/data') / dataset / 'CTC' / args.split
delta_path = Path('/home/max/MaxWork/DeLTA_TF_New')
res_path = delta_path / 'results' / dataset / args.split
res_path.mkdir(parents=True,exist_ok=True)

# Load config ('2D' or 'MOTHERMACHINE'):
if dataset == 'moma':
    config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE
elif dataset == 'DynamicNuclearNet-tracking-v1_0':
    config = delta.config.DEFAULT_CONFIG_2D
    config.target_size_seg = (256,256)
    config.target_size_track = (256,256)
elif dataset == 'My2D':
    config = delta.config.DEFAULT_CONFIG_2D
    config.target_size_seg = (256,256)
    config.target_size_track = (256,256)

config.model_file_seg = delta_path / 'models' / dataset / 'seg' / 'my2d_seg_model.keras'
config.model_file_track = delta_path / 'models' / dataset / 'track' / 'my2d_track_model.keras'
config.number_of_cores = 2
config.apply_tensorflow_config()
config.rotation_correction = False
config.drift_correction = False
config.models=("seg", "track")
config.save_format=("CTC",)

for save_format in config.save_format:
    (res_path / save_format).mkdir(exist_ok=True)

start_time = time.time()
total_frames = 0

pos_paths = sorted([pos_path for pos_path in datapath.iterdir() if re.findall('\d\d$',pos_path.name)])
for pos_path in pos_paths:
    file_path = datapath / '{p}' / 't{t}.tif'

    # Init reader (use bioformats=True if working with nd2, czi, ome-tiff etc):
    xpreader = delta.utils.XPReader(file_path,pos=pos_path.name)
    total_frames += len(xpreader.frames)
    gt_path = pos_path.parent / f"{pos_path.name}_GT" / "SEG"

    # Init pipeline:
    xp = delta.pipeline.Pipeline(xpreader, config=config, resfolder=res_path, gtfolder=gt_path)

    # Run it (you can specify which positions, which frames to run etc):
    xp.process()

    print("has seg_time:", hasattr(xp, "seg_time"))
    print("has track_time:", hasattr(xp, "track_time"))
    print("has total_time:", hasattr(xp, "total_time"))

total_time = time.time() - start_time
fps = total_frames / total_time

with open(str(res_path / 'FPS.txt'), 'w') as file:
    file.write(f"Frames per second (FPS): {fps:2f}\n")

print(round((time.time() - start_time)/60,2))