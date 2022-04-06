#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:47:07 2022

@author: ooconnor
"""

import os
import glob

import numpy as np
import cv2
    
import pytest

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape, predictCompilefromseg_track, saveResult_track
from delta.model import unet_seg, unet_track
import delta.utilities as utils
from delta.utilities import cfg
    
@pytest.mark.order(3)
def test_tracking_script_2D():
    _tracking_script('2D')
    
def test_tracking_script_moma():
    _tracking_script('mothermachine')
    
def _tracking_script(presets,batch_size=1):

    # Load config:
    cfg.load_config(presets=presets)

    # Input image sequences (change to whatever images sequence you want to evaluate):
    inputs_folder = cfg.eval_movie + '/cropped_rois' if presets == 'mothermachine' else cfg.eval_movie
    segmentation_folder = os.path.join(inputs_folder, "segmentation")

    # Output folder:
    outputs_folder = os.path.join(inputs_folder, "tracking")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    # List files to read
    unprocessed = sorted(
        glob.glob(inputs_folder + "/*.tif") + glob.glob(inputs_folder + "/*.png")
    )

    # Get original image size:
    imsize = cv2.imread(unprocessed[0], cv2.IMREAD_GRAYSCALE).shape

    # Load up model:
    model = unet_track(input_size=cfg.target_size_track + (4,))
    model.load_weights(cfg.model_file_track)

    # Process
    while unprocessed:
        # Pop out filenames
        ps = min(5, len(unprocessed))
        to_process = unprocessed[0:ps]
        del unprocessed[0:ps]

        print("\n###### Now processing: ")
        for file in to_process:
            print(file)

        # Get data:
        inputs, seg_filenames, boxes = predictCompilefromseg_track(
            inputs_folder,
            segmentation_folder,
            files_list=to_process,
            target_size=cfg.target_size_track,
            crop=cfg.crop_windows,
        )

        print("Cells to track: %d" % len(inputs))

        # Predict (have to do it in batches otherwise run into memry issues):
        results = np.empty_like(inputs, shape=inputs.shape[:3] + (1,))
        for i in range(0, len(inputs), 32):
            j = min((len(inputs), i + 32))
            results[i:j] = model.predict(inputs[i:j], verbose=1, batch_size=1)

        # Paste results into masks of original imsize:
        if cfg.crop_windows:
            _results = np.zeros_like(results, shape=(len(results),) + imsize + (1,))
            for r, mask in enumerate(results):
                cb, fb = boxes[r]
                _results[r, cb["ytl"] : cb["ybr"], cb["xtl"] : cb["xbr"]] = results[
                    r, fb["ytl"] : fb["ybr"], fb["xtl"] : fb["xbr"]
                ]
            results = _results

        # Save (use the filenames list from the data compiler)
        saveResult_track(outputs_folder, results, files_list=seg_filenames)


@pytest.mark.order(2)
def test_segmentation_script_2D():
    _segmentation_script('2D')
    
def test_segmentation_script_moma():
    _segmentation_script('mothermachine')

def _segmentation_script(presets):
      
    # Load config:
    cfg.load_config(presets=presets)
    
    # Input image sequence (change to whatever images sequence you want to evaluate):
    inputs_folder = cfg.eval_movie + '/cropped_rois' if presets == 'mothermachine' else cfg.eval_movie
    
    # Outputs folder:
    outputs_folder = os.path.join(inputs_folder, "segmentation")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
    
    # List files in inputs folder:
    unprocessed = sorted(
        glob.glob(inputs_folder + "/*.tif") + glob.glob(inputs_folder + "/*.png")
    )
    
    # Load up model:
    model = unet_seg(input_size=cfg.target_size_seg + (1,))
    model.load_weights(cfg.model_file_seg)
    
    # Process
    while unprocessed:
        # Pop out filenames
        ps = min(4096, len(unprocessed))  # 4096 at a time
        to_process = unprocessed[0:ps]
        del unprocessed[0:ps]
    
        # Input data generator:
        predGene = predictGenerator_seg(
            inputs_folder,
            files_list=to_process,
            target_size=cfg.target_size_seg,
            crop=cfg.crop_windows,
        )
    
        # mother machine: Don't crop images into windows
        if not cfg.crop_windows:
            # Predictions:
            results = model.predict(predGene, verbose=1)[:, :, :, 0]
    
        # 2D: Cut into overlapping windows
        else:
            img = readreshape(
                os.path.join(inputs_folder, to_process[0]),
                target_size=cfg.target_size_seg,
                crop=True,
            )
            # Create array to store predictions
            results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))
            # Crop, segment, stitch and store predictions in results
            for i in range(len(to_process)):
                # Crop each frame into overlapping windows:
                windows, loc_y, loc_x = utils.create_windows(
                    next(predGene)[0, :, :], target_size=cfg.target_size_seg
                )
                # We have to play around with tensor dimensions to conform to
                # tensorflow's functions:
                windows = windows[:, :, :, np.newaxis]
                # Predictions:
                pred = model.predict(windows, verbose=1, steps=windows.shape[0])
                # Stich prediction frames back together:
                pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
                pred = pred[np.newaxis, :, :, np.newaxis]  # Mess around with dims
    
                results[i] = pred
    
        # Post process results (binarize + light morphology-based cleaning):
        results = postprocess(results, crop=cfg.crop_windows)
    
        # Save to disk:
        saveResult_seg(outputs_folder, results, files_list=to_process)