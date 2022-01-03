#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:01:05 2021

@author: ooconnor
"""
#
import random

import pytest
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

from delta import utils

#%% Eval create_windows


def rand_img(image_size):
    img = np.random.random(image_size)
    return img

@pytest.mark.order(2)
class Test_create_windows:
    def test_create_windows_base(self):
        assert (
            utils.create_windows(
                rand_img((512, 512)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 1
        )

    def test_create_windows_2048x2048(self):
        assert (
            utils.create_windows(
                rand_img((2048, 2048)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 25
        )

    def test_create_windows_200x200(self):
        assert (
            utils.create_windows(
                rand_img((200, 200)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 1
        )

    def test_create_windows_200x600(self):
        assert (
            utils.create_windows(
                rand_img((200, 600)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 2
        )

    def test_create_windows_600x200(self):
        assert (
            utils.create_windows(
                rand_img((600, 200)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 2
        )

    def test_create_windows_100x700(self):
        assert (
            utils.create_windows(
                rand_img((1000, 700)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 6
        )

    def test_create_windows_696x520(self):
        assert (
            utils.create_windows(
                rand_img((696, 520)), target_size=(512, 512), min_overlap=24
            )[0].shape[0]
            == 4
        )


#%% Eval find_contours


def rand_mask(image_size, max_dots=100, max_radius=30):
    "Generate a random binary mask with random size dots"
    mask = np.zeros(image_size, dtype=np.uint8)
    num_dots = random.randint(0, max_dots)

    for dot in range(num_dots):
        cv2.circle(
            mask,
            center=(random.randint(0, image_size[0]), random.randint(0, image_size[0])),
            radius=random.randint(1, max_radius),
            color=1,
            thickness=-1,
        )

    # Holes cause issues here
    mask = binary_fill_holes(mask).astype(np.uint8)

    return mask


def evaluate_contours(mask, contours, image_size):
    "Reconstruct mask from contours and make it is the same as the original"

    new_mask = np.zeros(image_size, dtype=np.uint8)
    new_mask = cv2.drawContours(new_mask, contours, -1, 1, thickness=-1)

    return not (np.bitwise_xor(mask, new_mask).any())


@pytest.mark.order(2)
@pytest.mark.parametrize("execution_number", range(10))  # Run it 10 times
class Test_find_contours:
    def test_find_contours_base(self, execution_number):
        image_size = (512, 512)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_2048x2048(self, execution_number):
        image_size = (2048, 2048)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_200x200(self, execution_number):
        image_size = (200, 600)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_200x600(self, execution_number):
        image_size = (200, 600)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_600x200(self, execution_number):
        image_size = (600, 200)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_100x700(self, execution_number):
        image_size = (100, 700)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_696x520(self, execution_number):
        image_size = (696, 520)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)

    def test_find_contours_256x32(self, execution_number):
        image_size = (256, 32)
        mask = rand_mask(image_size)
        contours = utils.find_contours(mask)
        assert evaluate_contours(mask, contours, image_size)


def test_skeleton_poles():
    skel = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0],
                ],
            dtype=np.bool_)
    mask_poles = utils.skeleton_poles(skel)
    solution = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        ], dtype=np.bool_)
    assert np.array_equal(mask_poles, solution)


def test_extract_poles():
    labels = np.array(
            [
                [1, 1, 0, 2, 2, 2],
                [1, 1, 0, 2, 2, 2],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 4],
                [3, 3, 3, 0, 4, 4],
                ])
    skel = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 1],
                ],
            dtype=np.bool_)
    mask_poles = utils.skeleton_poles(skel)
    poles = utils.extract_poles(mask_poles, labels)
    solution = [
            [np.array([0, 1]), np.array([2, 1])],   # Cell 1
            [np.array([0, 3]), np.array([1, 5])],   # Cell 2
            [np.array([4, 0]), np.array([4, 2])],   # Cell 3
            [np.array([3, 4]), np.array([4, 5])],   # Cell 4
            ]
    assert np.array_equal(poles, solution)

#%% Test xpreader


from pathlib import Path
import sys
import delta

class Test_xpreader():

    def test_2D_tif_cmdln_filename_noresfolder(self):
    # 'Pretend' arguments were passed to the cmd line:
        sys.argv = [sys.argv[0]]
        sys.argv.append(Path(__file__).parent / 'data' / 'movie_2D_tif')
    
        # Load configuration:
        delta.utils.cfg.load_config(presets="2D", config_level="global")
    
        # Init reader
        xpreader = delta.utils.xpreader()
    
        # Init pipeline:
        xp = delta.pipeline.Pipeline(xpreader)

    def test_2D_tif_cmdln_allargs(self):
    # 'Pretend' arguments were passed to the cmd line:
        sys.argv = [sys.argv[0]]
        sys.argv.append(Path(__file__).parent / 'data' / 'movie_2D_tif')
        
        sys.argv.append('--resfolder')
        sys.argv.append(Path(__file__).parent / 'data' / 'movie_2D_tif' / 'results')
        
        sys.argv.append('--order')
        sys.argv.append('pct')
        
        sys.argv.append('--index')
        sys.argv.append(1)
    
        sys.argv.append('--prototype')
        sys.argv.append('Position%02dChanel%02dFrame%06d.tif')  # Currently there is a typo where it is chanel not channel

        # Load configuration:
        delta.utils.cfg.load_config(presets="2D", config_level="global")
    
        # Init reader
        xpreader = delta.utils.xpreader()
    
        # Init pipeline:
        xp = delta.pipeline.Pipeline(xpreader)
        
    def test_2D_tif_xpreader_filename_noresfolder(self):
        
        sys.argv = [sys.argv[0]]
  
        # Load configuration:
        delta.utils.cfg.load_config(presets="2D", config_level="global")
    
        # Init reader
        xpreader = delta.utils.xpreader(Path(__file__).parent / 'data' / 'movie_2D_tif')
    
        # Init pipeline:
        xp = delta.pipeline.Pipeline(xpreader)
        
    def test_2D_tif_xpreader_filename_and_resfolder(self):
        
        sys.argv = [sys.argv[0]]
  
        # Load configuration:
        delta.utils.cfg.load_config(presets="2D", config_level="global")
    
        # Init reader
        xpreader = delta.utils.xpreader(Path(__file__).parent / 'data' / 'movie_2D_tif')
    
        # Init pipeline:
        xp = delta.pipeline.Pipeline(
            xpreader,
            resfolder = Path(__file__).parent / 'data' / 'movie_2D_tif' / 'results'
            )
    
    def test_moma_tif_xpreader_filename_noresfolder(self):
        
        sys.argv = [sys.argv[0]]
  
        # Load configuration:
        delta.utils.cfg.load_config(presets="2D", config_level="global")
    
        # Init reader
        xpreader = delta.utils.xpreader(Path(__file__).parent / 'data' / 'movie_mothermachine_tif')
    
        # Init pipeline:
        xp = delta.pipeline.Pipeline(xpreader)
        
        