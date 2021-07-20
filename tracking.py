'''
This script runs the tracking U-Net on data out of the segmentation U-Net.

Images are processed in batches of 512, although the number of actual samples
run through the tracking U-Net will depend on the number of cells in each 
image.

@author: jblugagne
'''
import config as cfg

from data import saveResult_track, predictCompilefromseg_track
from model import unet_track
from os import listdir

# Files:
images_folder = 'D:/DeLTA_data/mother_machine/evaluation/preprocessed/img/'
segmentation_folder = 'D:/DeLTA_data/mother_machine/evaluation/seg_output/'
outputs_folder = 'D:/DeLTA_data/mother_machine/evaluation/track_output/'
unprocessed = listdir(images_folder)

# Load up model:
model = unet_track(input_size = cfg.target_size_track+(4,))
model.load_weights(cfg.model_file_track)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(512,len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Get data:
    inputs, seg_filenames = predictCompilefromseg_track(
        images_folder, 
        segmentation_folder,
        files_list = to_process, 
        target_size = cfg.target_size_track
        )
    
    # Predict:
    results = model.predict(inputs,verbose=1)
    
    # Save (use the filenames list from the data compiler)
    saveResult_track(outputs_folder,results, files_list = seg_filenames)