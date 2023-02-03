"""
Run the rois identification/segmentation U-Net.

To see how to extract roi images with this segmentation mask, see the
preprocessing section of pipeline.py and getROIBoxes() in utilities.py

@author: jblugagne
"""

import delta
from delta.data import postprocess, predictGenerator_seg, saveResult_seg
from delta.utilities import listfiles

### Parameters:

# Default config
config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml").load()

# Image sequence to segment: (replace with your own)
inputs_folder = delta.config._DELTA_DIR.parent / "tests/data/movie_mothermachine_tif"

### /Parameters


# Output folder:
outputs_folder = inputs_folder / "roi_masks"
outputs_folder.mkdir(exist_ok=True)

# List files in inputs folder:
input_files = listfiles(inputs_folder, {".tif", ".png"})

# Load up model:
model = config.model("rois")

# Inputs data generator:
predGene = predictGenerator_seg(
    inputs_folder, files_list=input_files, target_size=config.target_size_rois
)

# Predictions:
results = model.predict(predGene, len(input_files), verbose=1)

# Post process results:
results[:, :, :, 0] = postprocess(results[:, :, :, 0])

# Save to disk:
saveResult_seg(outputs_folder, results, files_list=input_files)
