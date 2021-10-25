#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:01:05 2021

@author: ooconnor
"""
#
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','delta')))

import utilities as utils
import pytest
import numpy as np

def rand_img(image_size):
    img = np.random.random(image_size)
    return img

class Test_create_windows:
    def test_create_windows_base(self):
        assert utils.create_windows(rand_img((512,512)),target_size=(512,512),min_overlap=24)[0].shape[0] == 1
    def test_create_windows_2048x2048(self):
        assert utils.create_windows(rand_img((2048,2048)),target_size=(512,512),min_overlap=24)[0].shape[0] == 25
    def test_create_windows_200x200(self):
        assert utils.create_windows(rand_img((200,200)),target_size=(512,512),min_overlap=24)[0].shape[0] == 1
    def test_create_windows_200x600(self):
        assert utils.create_windows(rand_img((200,600)),target_size=(512,512),min_overlap=24)[0].shape[0] == 2
    def test_create_windows_600x200(self):
        assert utils.create_windows(rand_img((600,200)),target_size=(512,512),min_overlap=24)[0].shape[0] == 2
    def test_create_windows_100x700(self):
        assert utils.create_windows(rand_img((1000,700)),target_size=(512,512),min_overlap=24)[0].shape[0] == 6
    def test_create_windows_696x520(self):
        assert utils.create_windows(rand_img((696,520)),target_size=(512,512),min_overlap=24)[0].shape[0] == 4





