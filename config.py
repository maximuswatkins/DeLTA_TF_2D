# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:42:05 2021

@author: jblugagne
"""

# If running into OOM issues or having trouble with cuDNN loading, try setting
# memory_growth_limit to a value in MB: (eg 1024, 2048...)
memory_growth_limit = None

if memory_growth_limit is not None:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


# Download latest models first:
# https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP

# Path to model files:
model_file_chambers = 'C:/DeepLearning/DeLTA_data/mother_machine/models/chambers_id_tessiechamp.hdf5'
model_file_seg = 'C:/DeepLearning/DeLTA_data/mother_machine/models/unet_moma_seg_multisets.hdf5'
model_file_track = 'C:/DeepLearning/DeLTA_data/mother_machine/models/unet_moma_track_multisets.hdf5'

# Model parameters:
target_size_chambers = (512, 512)
input_size_chambers = target_size_chambers + (1,)
target_size_seg = (256, 32)

# Misc:
min_chamber_area = 500 # Minimum size of chambers in pixels for area filtering
