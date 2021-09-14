# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:42:05 2021

@author: jblugagne
"""

# Download latest models first:
# https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP

#%% Functions used to custom config.py from home directory if available
import os 
    
def import_module_by_path(path):
    '''
    
    Import a module from a different directory than the current working directory
    
    Parameters
    ----------
    path : str
        Path to local config.py file.

    Returns
    -------
    mod : module
        Returns the config.py file as a module.

    '''
    name = os.path.splitext(os.path.basename(path))[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod

def import_star_module(mod):
    '''
    This function performs: from mod import *
    It imports all names directly into the local namespace

    Parameters
    ----------
    mod : module
        config.py as a module.

    Returns
    -------
    None.

    '''
    if "__all__" in mod.__dict__:
        names = mod.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mod.__dict__ if not x.startswith("_")]
        
    globals().update({k: getattr(mod, k) for k in names})

#%%

# Customized config.py should be stored in your home directory within a delta folder
# If there is no config.py at this location, the local config.py will be used
path = os.path.expanduser('~/delta/config.py')

# To prevent recursive checking for the config.py within the home dirctory,
# the and statement checks if the config.py exists and checks which directory
# the script is currently running in
if os.path.exists(path) and os.path.realpath(__file__) != path:
    mod = import_module_by_path(path)
    import_star_module(mod)
else: 
    presets = '2D' # 'mothermachine' or '2D'
    
    if presets == 'mothermachine':
        models = ['rois', 'segmentation', 'tracking']
        model_file_rois = 'D:/DeLTA_data/mother_machine/models/chambers_id_tessiechamp.hdf5'
        model_file_seg = 'D:/DeLTA_data/mother_machine/models/unet_moma_seg_multisets.hdf5'
        model_file_track = 'D:/DeLTA_data/mother_machine/models/unet_moma_track_v2.hdf5'
        target_size_rois = (512, 512)
        target_size_seg = (256, 32)
        target_size_track = target_size_seg
        
        # Training sets:
        training_set_rois = 'D:/DeLTA_data/mother_machine/training/chambers_seg_set/train'
        training_set_seg = 'D:/DeLTA_data/mother_machine/training/segmentation_set/train_multisets/'
        training_set_track = 'D:/DeLTA_data/mother_machine/training/tracking_set/train_multisets'
        
        # Pipeline parameters:
        rotation_correction = True # Whether to perform automated image rotation correction
        drift_correction = True # Whether to perform image drift correction
        crop_windows = False # Whether to crop out windows for segmentation and tracking
        min_roi_area = 500 # Minimum size of ROIs/chambers in pixels for area filtering (0 to +Inf)
        
    elif presets == '2D':
        models = ['segmentation', 'tracking']
        model_file_seg = 'D:/DeLTA_data/agar_pads/unet_pads_seg.hdf5'
        model_file_track = 'D:/DeLTA_data/agar_pads/unet_pads_track.hdf5'
        target_size_seg = (512, 512)
        target_size_track = (256, 256)
        
        # Training sets:
        training_set_seg = 'D:/DeLTA_data/agar_pads/trainingset_seg_20210304/'
        training_set_track = 'D:/DeLTA_data/agar_pads/tracking_set'
        
        # Pipeline parameters:
        rotation_correction = False # Whether to perform automated image rotation correction
        drift_correction = False # Whether to perform image drift correction
        crop_windows = True # Whether to crop out windows for segmentation and tracking
    
    # Other pipeline parameters:
    whole_frame_drift = False # Whether to use the whole frame for drift correction
    min_cell_area = 20 # Minimum area of cells in pixels
    save_format = ('pickle', 'legacy', 'movie') # Format(s)) to save the data to.
    
    #%% Tensorflow technical parameters:
    # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
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
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_growth_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)