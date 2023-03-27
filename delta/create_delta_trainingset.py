from pathlib import Path
import cv2
import shutil
import numpy as np

from delta import data

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma')
delta_datapath = datapath / 'delta'
delta_datapath.mkdir(exist_ok=True)
(delta_datapath / 'segmentation_set').mkdir(exist_ok=True)

for train_folder in ['train','val']:

    (delta_datapath / 'segmentation_set' / train_folder).mkdir(exist_ok=True)

    for folder in ['img','seg']:
        (delta_datapath / 'segmentation_set' / train_folder / folder).mkdir(exist_ok=True)

    fps = sorted(list((datapath / train_folder / 'img').glob('*.png'))) + sorted(list((datapath / train_folder / 'img').glob('*.tif')))

    for fp in fps:

        img = shutil.copyfile(datapath  / train_folder / 'img' / fp.name, delta_datapath / 'segmentation_set' / train_folder / 'img' / fp.name)

        outputs = cv2.imread(str(datapath / train_folder / 'gt' / fp.name),cv2.IMREAD_ANYDEPTH)

        seg = ((outputs > 0) * 255).astype(np.uint8)

        cv2.imwrite(str(delta_datapath / 'segmentation_set' / train_folder / 'seg' / fp.name), seg)

    data.make_weights(delta_datapath / 'segmentation_set' / train_folder,data.seg_weights)