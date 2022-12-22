"""
This script will run the rois identification/segmentation U-Net. 

To see how to extract roi images with this segmentation masks, see the 
preprocessing section of pipeline.py and getROIboxes() in utilities.py

@author: jblugagne
"""

from delta.assets import demo_movie_path, loadmodels
from delta.data import postprocess, predictGenerator_seg, saveResult_seg
from delta.model import unet_rois
from delta.utilities import cfg, listfiles

### Parameters:

# Load config file:
cfg.load_config(presets="mothermachine")

# Images sequence: (replace with your own)
inputs_folder = demo_movie_path()

### /Parameters


# Output folder:
outputs_folder = inputs_folder / "roi_masks"
outputs_folder.mkdir(exist_ok=True)

# List files in inputs folder:
input_files = listfiles(inputs_folder, {".tif", ".png"})

# Load up model:
model = loadmodels()["rois"]

# Inputs data generator:
predGene = predictGenerator_seg(
    inputs_folder, files_list=input_files, target_size=cfg.target_size_rois
)

# Predictions:
results = model.predict(predGene, len(input_files), verbose=1)

# Post process results:
results[:, :, :, 0] = postprocess(results[:, :, :, 0])

# Save to disk:
saveResult_seg(outputs_folder, results, files_list=input_files)
