'''
This script runs the segmentation U-Net on data that has been pre-processed to
crop out chamber images. See delta-interfacing/preprocessing.m or pipeline.py.

The images are processed by batches of 4096 to prevent memory issues.

@author: jblugagne
'''
import config as cfg

from data import saveResult_seg, predictGenerator_seg, postprocess
from model import unet_seg
import os

# Files:
inputs_folder = 'D:/DeLTA_data/mother_machine/evaluation/preprocessed/img/'
outputs_folder = 'D:/DeLTA_data/mother_machine/evaluation/seg_output/'
unprocessed = os.listdir(inputs_folder)

# Load up model:
model = unet_seg(input_size = cfg.target_size_seg+(1,))
model.load_weights(cfg.model_file_seg)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(4096,len(unprocessed)) #4096 at a time
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Predict:
    predGene = predictGenerator_seg(
        inputs_folder, 
        files_list = to_process,
        target_size = cfg.target_size_seg
        )
    results = model.predict(predGene,verbose=1)
    
    # Post process results:
    results[:,:,:,0] = postprocess(results[:,:,:,0])
    
    # Save to disk:
    saveResult_seg(outputs_folder,results, files_list = to_process)