"""
Utility functions and class definitions that are used in pipeline.py.

A few functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

@author: jblugagne
"""

import datetime
import re
import subprocess
import sys
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from tkinter import filedialog
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import bioformats
import cv2
import ffmpeg
import javabridge
import numpy as np
import numpy.random as npr
import numpy.typing as npt
from scipy.special import logit
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks
from tensorflow.keras.callbacks import TensorBoard
from tifffile import TiffFile

from delta.lineage import CellFeatures, Pole

Image = npt.NDArray[np.float32]

Contour = np.ndarray

Labels = npt.NDArray[np.uint16]

SegmentationMask = npt.NDArray[np.uint8]


@dataclass
class CroppingBox:
    """
    Class describing a box to cut out.

    Attributes
    ----------
        xtl : int
            Top-left corner X coordinate.
        ytl : int
            Top-left corner Y coordinate.
        xbr : int
            Bottom-right corner X coordinate.
        ybr : int
            Bottom-right corner Y coordinate.
    """

    xtl: int
    ytl: int
    xbr: int
    ybr: int

    @staticmethod
    def full(image: np.ndarray):
        """
        Return a cropping box set to the full size of the image.

        Arguments
        ---------
            image : np.ndarray
                Image to use as reference for the bounding box.

        Returns
        -------
            box : CroppingBox
                Cropping box adjusted to the full size of the image.
        """
        return CroppingBox(xtl=0, ytl=0, xbr=image.shape[1], ybr=image.shape[0])

    def crop(self, image: np.ndarray):
        """
        Crop an image according to the cropping box.

        Pads with zeros if a part of the box falls outside of the image.

        Arguments
        ---------
            image : np.ndarray
                Image to crop.

        Returns
        -------
            patch : np.ndarray
                Patch cropped from the image.
        """
        if image.ndim != 2:
            raise ValueError("`image` must have 2 dimensions.")
        cropped = image[
            max(self.ytl, 0) : min(self.ybr, image.shape[0]),
            max(self.xtl, 0) : min(self.xbr, image.shape[1]),
        ]
        padding = (
            (max(-self.ytl, 0), max(self.ybr - image.shape[0], 0)),
            (max(-self.xtl, 0), max(self.xbr - image.shape[1], 0)),
        )
        return np.pad(cropped, padding)

    def patch(self, image: np.ndarray, patch: np.ndarray):
        """
        Apply a patch on an image at the position specified by the box.

        Parts of the box may fall outside of the image.

        Arguments
        ---------
            image : np.ndarray
                Image to patch.
            patch : np.ndarray
                Patch to apply.

        Returns
        -------
            image : np.ndarray
                The patched image.
        """
        if image.ndim != 2 or patch.ndim != 2:
            raise ValueError("`image` and `patch` must have 2 dimensions.")
        if patch.shape != (self.ybr - self.ytl, self.xbr - self.xtl):
            raise ValueError(
                "`patch` must have the same dimensions as the cropping box."
            )
        if (
            self.ybr <= 0
            or image.shape[1] <= self.ytl
            or self.xbr <= 0
            or image.shape[0] <= self.xtl
        ):
            raise ValueError("`box` must fall at least partially inside the image.")
        image[
            max(self.ytl, 0) : min(self.ybr, image.shape[0]),
            max(self.xtl, 0) : min(self.xbr, image.shape[1]),
        ] = patch[
            max(-self.ytl, 0) : min(self.ybr, image.shape[0]) - self.ytl,
            max(-self.xtl, 0) : min(self.xbr, image.shape[1]) - self.xtl,
        ]
        return image


class Cell(Dict):
    """
    Dictionary describing a cell, its lineage and pole information.

    Attributes
    ----------
        id : int
            Cell ID number.
        mother : Optional[int]
           ID of mother cell. If None, the cell is an orphan.
        frames : List[int]
            List of frames / timepoints where the cell is present.
        daughters : List[Optional[int]]
            List of daughters over time. If None, no division occurred.
        new_pole : List[Optional[Pole]]
            List of identified new Pole positions over time.
        old_pole : List[Optional[Pole]]
            List of identified old Pole positions over time.
    """

    id: int
    mother: Optional[int]
    frames: List[int]
    daughters: List[Optional[int]]
    new_pole: List[Optional[Pole]]
    old_pole: List[Optional[Pole]]

    def __eq__(self, other) -> bool:
        """Equality function for Cell."""
        if "fluo" not in other:
            if any(key.startswith("fluo") for key in other.keys()):
                fluo = np.vstack(
                    [
                        fluo
                        for key, fluo in sorted(other.items())
                        if key.startswith("fluo")
                    ]
                ).T
            else:
                fluo = np.empty((len(self["frames"]), 0))
            for key in list(other.keys()):
                if key.startswith("fluo"):
                    del other[key]
            other["fluo"] = fluo
        if self.keys() != other.keys():
            print("Cells differ in attributes:")
            print(f"    {list(self.keys())}")
            print(f"    {list(other.keys())}")
            return False
        for key, value in self.items():
            try:
                try:
                    np.testing.assert_allclose(value, other[key], atol=1e-8)
                except TypeError:
                    assert value == other[key]
            except AssertionError as err:
                print(f"Cells differ by {key}:")
                print(err)
                return False
        return True


# %% Image correction


def to_integer_values(frame: Image, dtype: Type[object]) -> Image:
    """
    Return an image with integer-valued pixels of the given size.

    Parameters
    ----------
    frame : npt.NDArray[np.float32]
        Input image with np.float32 pixels contained in [0, 1].
    dtype : numpy dtype
        Integer type of the returned image.


    """
    assert frame.max() <= 1, "The pixel values should be contained between 0 and 1."
    depth = {np.uint8: 8, np.uint16: 16, np.uint32: 32}[dtype]
    return (frame * (2**depth - 1)).astype(dtype)


def deskew(image: Image) -> float:
    """
    Compute the rotation angle to apply to the image to remove its rotation.

    You can skip rotation correction if your chambers are about +/- 1 degrees of horizontal.

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    angle : float
        Rotation angle of the chambers for correction, in degrees.

    """
    image8 = to_integer_values(image, np.uint8)

    # enhance edges
    low_threshold = np.quantile(image8, 0.1)
    high_threshold = np.quantile(image8, 0.2)
    edges = cv2.Canny(image8, low_threshold, high_threshold, L2gradient=True)

    # Hough transform
    N = 360  # precision in degree = 90 / N (here 0.25 degree)
    theta = np.linspace(-np.pi / 4, 3 * np.pi / 4, 2 * N + 1)
    hspace, angles, distances = hough_line(edges, theta=theta)

    _, corrections, _ = hough_line_peaks(hspace, angles, distances, num_peaks=1)

    if corrections[0] > np.pi / 4:
        return np.degrees(corrections[0] - np.pi / 2)
    return np.degrees(corrections[0])


def imrotate(frame: npt.NDArray[Any], rotate: float) -> npt.NDArray[Any]:
    """
    Rotate image.

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rotate : float
        Rotation angle, in degrees.

    Returns
    -------
    2D numpy array of floats
        Rotated image

    """
    M = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), rotate, 1)
    frame = cv2.warpAffine(
        frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )

    return frame


# TODO: refactor this (maybe separate drift computation and application)
def driftcorr(
    img: Image,
    template: Optional[Image] = None,
    box: Optional[CroppingBox] = None,
    drift: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Image, Union[Tuple[int, int], Tuple[List[int], List[int]]]]:
    """
    Compute and correct drift between current frame and the reference.

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drfit for.
    template : 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        Default is None.
    box : CroppingBox, optional
        A cropping box to extract the part of the frame to compute drift
        correction over.
        Default is None.
    drift : tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        Default is None.

    Returns
    -------
    image : 2D/3D numpy array
        Drift-corrected image.
    xcorr, ycorr : float, float
        Drift used.
    """
    if len(img.shape) == 2:
        twoDflag = True
        img = np.expand_dims(img, axis=0)
    else:
        twoDflag = False

    if drift is None:
        if template is None:
            # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0, 0)
        template = to_integer_values(template, np.uint8)
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]], [drift[1]])
    else:
        (xcorr, ycorr) = drift

    for i in range(img.shape[0]):
        if drift is None:
            if box is None:
                raise ValueError("If `drift` is None, a box must be provided.")
            frame = to_integer_values(img[i], np.uint8)
            driftcorrimg = box.crop(frame)
            res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            ycorr[i] = max_loc[0] - res.shape[1] / 2
            xcorr[i] = max_loc[1] - res.shape[0] / 2
        T = np.array([[1, 0, -ycorr[i]], [0, 1, -xcorr[i]]], dtype=np.float32)
        img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])

    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    return img, (xcorr, ycorr)


def getDriftTemplate(
    chamberboxes: List[CroppingBox], img: Image, whole_frame: bool = False
) -> Optional[Image]:
    """
    Retrieve a region above the chambers to use as drift template.

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getROIBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.
    whole_frame : bool, optional
        Whether to use the whole frame as reference instead of the area above
        the chambers.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    """
    if len(chamberboxes) == 0 and not whole_frame:
        return None

    # Cutting out 2.5% of the image on each side as drift margin
    y_cut, x_cut = (round(i * 0.025) for i in img.shape)

    box = CroppingBox(
        xtl=x_cut,
        xbr=-x_cut,
        ytl=y_cut,
        ybr=-y_cut if whole_frame else max(box.ytl for box in chamberboxes) - y_cut,
    )

    return box.crop(img)


def opencv_areafilt(
    image: SegmentationMask,
    min_area: Optional[float] = 20,
    max_area: Optional[float] = None,
) -> SegmentationMask:
    """
    Area filtering using openCV instead of skimage.

    Parameters
    ----------
    image : 2D array
        Segmentation mask.
    min_area : float or None, optional
        Minimum object area.
        The default is 20
    max_area : float or None, optional
        Maximum object area.
        The default is None.

    Returns
    -------
    image : 2D array
        Filtered mask.

    """
    # Get contours:
    contours = find_contours(image)

    # Loop through contours, flag them for deletion:
    to_remove = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (min_area is not None and area < min_area) or (
            max_area is not None and area > max_area
        ):
            to_remove += [cnt]

    # Delete all at once:
    if len(to_remove) > 0:
        image = cv2.drawContours(image, to_remove, -1, 0, thickness=-1)

    return image


# %% Image cropping & stitching


def create_windows(
    image: npt.NDArray, target_size: Tuple[int, int] = (512, 512), min_overlap: int = 24
) -> Tuple[npt.NDArray, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Crop input image into windows of set size.

    Parameters
    ----------
    image : 2D array
        Input image.
    target_size : tuple, optional
        Dimensions of the windows to crop out. The default is (512, 512).
    min_overlap : int, optional
        Minimum overlap between windows in pixels. Default is 24.


    Returns
    -------
    windows: 3D array
        Cropped out images to feed into U-Net. Dimensions are
        (nb_of_windows, *target_size).
    loc_y : array of shape (nsplits_y, 2)
        List of lower and upper bounds for windows over the y axis (columns).
    loc_x : array of shape (nsplits_x, 2)
        List of lower and upper bounds for windows over the x axis (rows).

    """
    # Compute how many windows we will have
    stride_x = target_size[1] - min_overlap
    stride_y = target_size[0] - min_overlap
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("min_overlap must be strictly less than target_size")
    nsplits_x = 1 + int(np.ceil((image.shape[1] - target_size[1]) / stride_x))
    nsplits_y = 1 + int(np.ceil((image.shape[0] - target_size[0]) / stride_y))

    # Pad the image if smaller than the target size
    pad_x = max(0, target_size[1] - image.shape[1])
    pad_y = max(0, target_size[0] - image.shape[0])
    padding = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2))
    image = np.pad(image, padding)

    # Compute offsets
    starts_x = np.linspace(
        0, image.shape[1] - target_size[1], num=nsplits_x, endpoint=True, dtype=np.int32
    )
    starts_y = np.linspace(
        0, image.shape[0] - target_size[0], num=nsplits_y, endpoint=True, dtype=np.int32
    )

    # Store the cropped images
    tiles = np.array(
        [
            image[sy : sy + target_size[0], sx : sx + target_size[1]]
            for sy in starts_y
            for sx in starts_x
        ],
        dtype=image.dtype,
    )
    loc_x = np.vstack((starts_x, starts_x + target_size[1])) - pad_x // 2
    loc_y = np.vstack((starts_y, starts_y + target_size[0])) - pad_y // 2
    return tiles, loc_y.T, loc_x.T


def stitch_pic(
    results: npt.NDArray[Any],
    loc_y: List[Tuple[int, int]],
    loc_x: List[Tuple[int, int]],
) -> npt.NDArray[Any]:
    """
    Stitch segmentation back together from the windows of create_windows().

    Parameters
    ----------
    results : 3D array
        Segmentation outputs from the seg model with dimensions
        (nb_of_windows, target_size[0], target_size[1])
    loc_y : list
        List of lower and upper bounds for windows over the y axis
    loc_x : list
        List of lower and upper bounds for windows over the x axis

    Returns
    -------
    stitch_norm : 2D array
        Stitched image.

    """
    # Create an array to store segmentations into a format similar to how the image was cropped
    stitch = np.zeros((loc_y[-1][1], loc_x[-1][1]), dtype=results.dtype)

    # Compute where patches meet
    middles_y = [
        0,
        *((ly[1] + nly[0]) // 2 for ly, nly in zip(loc_y, loc_y[1:])),
        loc_y[-1][1],
    ]
    middles_x = [
        0,
        *((lx[1] + nlx[0]) // 2 for lx, nlx in zip(loc_x, loc_x[1:])),
        loc_x[-1][1],
    ]

    index = 0
    for ly, y_start, y_end in zip(loc_y, middles_y, middles_y[1:]):
        for lx, x_start, x_end in zip(loc_x, middles_x, middles_x[1:]):
            res_crop_y = -(ly[1] - y_end) if ly[1] - y_end > 0 else None
            res_crop_x = -(lx[1] - x_end) if lx[1] - x_end > 0 else None
            stitch[y_start:y_end, x_start:x_end] = results[
                index,
                y_start - ly[0] : res_crop_y,
                x_start - lx[0] : res_crop_x,
            ]

            index += 1

    return stitch


def gettrackingboxes(
    cell: npt.NDArray[np.uint8],
    frame_shape: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[CroppingBox, CroppingBox]:
    """
    Get a crop box and a fill box around a cell that fits the tracking target size.

    Parameters
    ----------
    cell : 2D array of uint8
        Mask of the cell to track.
    frame_shape : tuple of 2 ints
        Original dimensions of the image.
    target_size : tuple of 2 ints
        Target dimensions of the cropped image.

    Returns
    -------
    cropbox : CroppingBox
        The crop box determines which part of the full-size frame to crop out.
    fillbox : CroppingBox
        The fill box determines which part of the target-size input to fill with
        the cropped out image.

    """
    cx, cy = getcentroid(cell)

    xtl = int(max(cx - target_size[1] / 2, 0))
    xbr = int(min(cx + target_size[1] / 2, frame_shape[1]))

    ytl = int(max(cy - target_size[0] / 2, 0))
    ybr = int(min(cy + target_size[0] / 2, frame_shape[0]))

    cropbox = CroppingBox(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)

    xtl = int(max(target_size[1] / 2 - cx, 0))
    xbr = int(min(target_size[1] / 2 + frame_shape[1] - cx, target_size[1]))

    ytl = int(max(target_size[0] / 2 - cy, 0))
    ybr = int(min(target_size[0] / 2 + frame_shape[0] - cy, target_size[0]))

    fillbox = CroppingBox(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)

    return cropbox, fillbox


def getshiftvalues(
    shift: int, img_shape: Tuple[int, int], cropbox: CroppingBox, rng: npr.Generator
) -> Tuple[int, int]:
    """
    Get shift values.

    Parameters
    ----------
    shift : int
        Max amount of pixels cropbox to be shifted in the y / x direction.
    img_shape : tuple
        Shape of the image / input that will be cropped.
    cropbox : CroppingBox
        The crop box determines which part of the full-size frame to crop out.
    rng : npr.Generator
        Pseudo-random number generator.

    Returns
    -------
    shiftY : int
        Number of pixels to shift cropbox in the y direction
    shiftX : int
        Number of pixels to shift cropbox in the x direction.

    """
    upperY = np.min((shift, img_shape[0] - cropbox.ybr))
    lowerY = np.min((shift, cropbox.ytl))
    shiftY = round(rng.uniform(-lowerY, upperY))

    upperX = np.min((shift, img_shape[1] - cropbox.xbr))
    lowerX = np.min((shift, cropbox.xtl))
    shiftX = round(rng.uniform(-lowerX, upperX))

    return shiftY, shiftX


# %% Masks, labels, objects identification


def find_contours(mask: npt.NDArray[np.uint8]) -> List[Contour]:
    """
    Find contours of morphological components of a segmentation mask.

    This is a wrapper of CV2's findContours() because it keeps changing signatures.

    Parameters
    ----------
    mask : 2D array of uint8.
        Segmentation mask to extract contours from.

    Returns
    -------
    contours : list
        List of cv2 type contour arrays.

    """
    # Default use:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Some versions of cv2.findcontours return it as a tuple, not a list:
    contours = list(contours)

    return contours


def getROIBoxes(chambersmask: SegmentationMask) -> List[CroppingBox]:
    """
    Extract the bounding boxes of the chambers in the binary mask produced by the ROI segmentation model.

    Parameters
    ----------
    chambersmask : 2D array of uint8
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of `CroppingBox`es
        Chamber `CroppingBox`es.

    """
    chamberboxes = []

    contours = find_contours(chambersmask)
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamberboxes.append(
            # tl = top left, br = bottom right
            CroppingBox(
                xtl=xtl,
                # -10% of height to make sure the top is not cropped
                ytl=ytl - int(0.1 * boxheight),
                xbr=xtl + boxwidth,
                ybr=ytl + boxheight,
            )
        )

    # Sorting by top-left X (normally sorted by Y top left)
    chamberboxes.sort(key=lambda box: box.xtl)

    return chamberboxes


def label_seg(
    seg: SegmentationMask,
    cellnumbers: Optional[Sequence[int]] = None,
) -> Labels:
    """
    Label cells in segmentation mask.

    Parameters
    ----------
    seg : SegmentationMask
        Cells segmentation mask.
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    labels : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels
        with values given by cellnumbers

    """
    contours = find_contours(seg)
    if cellnumbers is None:
        cellnumbers = range(1, len(contours) + 1)
    if len(cellnumbers) != len(contours):
        raise ValueError("cellnumbers must have the same length as the number of cells")
    # Sorting according to the lowest point on the y (vertical) axis
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))
    labels = np.zeros(seg.shape, dtype=np.uint16)
    for icell, contour in zip(cellnumbers, contours):
        labels = cv2.fillPoly(labels, [contour], icell)
    return labels


@overload
def getcellsinframe(
    labels: Labels, return_contours: Literal[True]
) -> Tuple[List[int], List[Contour]]:
    ...


@overload
def getcellsinframe(
    labels: Labels, return_contours: Literal[False] = False
) -> List[int]:
    ...


def getcellsinframe(
    labels: Labels, return_contours: bool = False
) -> Union[List[int], Tuple[List[int], List[Contour]]]:
    """
    Get numbers of cells present in frame, sorted along Y axis.

    Parameters
    ----------
    labels : 2D numpy array of uint16
        Single frame from labels stack.
    return_contours : bool, optional
        Flag to get cv2 contours.

    Returns
    -------
    cellids : list
        Cell ids.
    contours : list
        List of cv2 contours for each cell. Returned if return_contours==True.

    """
    cellids, ind = np.unique(labels, return_index=True)

    # Sorting along Y axis & removing background
    cellids = [int(cellid) - 1 for cellid in cellids[1:][np.argsort(ind[1:])]]

    if not return_contours:
        return cellids

    # Get opencv contours:
    contours = [
        find_contours((labels == cellid + 1).astype(np.uint8))[0] for cellid in cellids
    ]

    return cellids, contours


def getcentroid(contour: np.ndarray) -> Tuple[int, int]:
    """
    Get centroid of cv2 contour.

    Parameters
    ----------
    contour : 3D numpy array
        Blob contour generated by cv2.findContours().

    Returns
    -------
    cx : int
        X-axis coordinate of centroid.
    cy : int
        Y-axis coordinate of centroid.

    """
    if contour.shape[0] > 2:  # Looks like cv2.moments treats it as an image
        # Calculate moments for each contour
        M = cv2.moments(contour)
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
    else:
        cx = int(np.mean(contour[:, :, 0]))
        cy = int(np.mean(contour[:, :, 1]))

    return cx, cy


# %% Poles


def getpoles(
    seg: SegmentationMask, labels: Labels, scaling: Optional[Tuple[float, float]] = None
) -> Dict[int, Tuple[Pole, Pole]]:
    """
    Get cell poles.

    Parameters
    ----------
    seg : 2D array of uint8
        Cell segmentation mask.
    labels : 2D array of int
        Cell labels map.
    scaling : tuple of floats, optional
        Scaling to apply in both directions to rescale poles' positions to
        original image size (if resizing was applied)

    Returns
    -------
    poles : Dict[int, (Pole, Pole)]
        Dictionary of cellid to poles. Each cell has exactly 2 poles.

    """
    # Init list of empty poles dictionaries:
    poles: Dict[int, List[Pole]] = {i - 1: [] for i in np.unique(labels)[1:]}

    # Get poles using skeleton method:
    skel = skeletonize(seg, method="lee")
    ends_map = skeleton_poles(skel)
    poles |= extract_poles(ends_map, labels)  # Set union with empty poles dict

    clean_poles = {}

    # Make sure cells have 2 poles each:
    for cellid, cellpoles in poles.items():
        if len(cellpoles) < 2:  # Sometimes skeletonize fails
            clean_poles[cellid] = extrema_poles(labels == cellid + 1, scaling=scaling)
        elif len(cellpoles) == 2:
            clean_poles[cellid] = tuple(cellpoles)
        else:
            clean_poles[cellid] = two_poles(cellpoles)

    if not scaling:
        return clean_poles

    # Apply image scaling to poles before returning:
    return {
        cellid: (
            np.round(p1 * scaling).astype(np.int16),
            np.round(p2 * scaling).astype(np.int16),
        )
        for cellid, (p1, p2) in clean_poles.items()
    }


def skeleton_poles(skel: SegmentationMask) -> SegmentationMask:
    """
    Filter out the poles of a skeleton.

    This function was adapted from stackoverflow:
        https://stackoverflow.com/q/26537313

    Parameters
    ----------
    skel : 2D numpy array of bool
        Contains skeletons of single cells from the segmentation.

    Returns
    -------
    out : 2D numpy array of bool
        Contains True at the poles of the skeletons from the input.

    """
    # create an integer array from the boolean one
    skel_uint8 = np.array(skel, dtype=np.uint8)

    # apply the convolution
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    ddepth = -1  # same depth as the source
    filtered = cv2.filter2D(skel_uint8, ddepth, kernel, borderType=cv2.BORDER_ISOLATED)

    # now look through to find the value of 11 (extremity) or 10 (single point)
    # this returns a mask of the poles
    return (filtered == 10) | (filtered == 11)


def extrema_poles(
    cellmask: SegmentationMask, scaling: Optional[Tuple[float, float]] = None
) -> Tuple[Pole, Pole]:
    """
    Find 2 poles per cell with the extrema method.

    A slower and more rudimentary version of poles finding but that is
    guaranteed to find exactly 2 poles.

    Parameters
    ----------
    cellmask : 2D array of int
        Segmentation mask of a single cell.
    scaling : Optional[Tuple[float, float]]
        Scaling to apply in both directions to rescale poles' positions to
        original image size (if resizing was applied).

    Returns
    -------
    poles : tuple of 2 1D arrays
        Two poles, each being a 2-element array of y-x coordinates.

    """
    scaling = scaling or (1, 1)

    y, x = cellmask.nonzero()
    if np.ptp(y) * scaling[0] > np.ptp(x) * scaling[1]:  # Vertical cell
        i = np.lexsort((x, y))
    else:  # Horizontal cell
        i = np.lexsort((y, x))
    poles = (
        np.array((y[i[0]], x[i[0]]), dtype=np.int16),
        np.array((y[i[-1]], x[i[-1]]), dtype=np.int16),
    )

    return poles


def two_poles(poles: List[Pole]) -> Tuple[Pole, Pole]:
    """
    Select only 2 poles out of possibly more than 2.

    Sometimes the skeleton produces more than 2 poles.
    Thus function selects only 2 poles in the skeleton.

    Parameters
    ----------
    poles : list
        Coordinates for the mother/daughter poles for a single cell
        Dimension is (#_of_poles, [y_coordinate, x_coordinate])


    Returns
    -------
    poles : tuple of 2 1D arrays
        Two poles, each being a 2-element array of y-x coordinates.

    """
    dist = 0.0
    # Measure the distance from the first endpoint to all the other poles
    for ipole, pole in enumerate(poles):
        d = eucl(poles[0], pole)
        if d > dist:
            dist = d
            i_max = ipole

    # Redo the same thing starting from the furthest endpoint:
    dist = 0.0
    for ipole, pole in enumerate(poles):
        if ipole != i_max:  # skip same point
            d = eucl(poles[i_max], pole)
            if d > dist:
                dist = d
                j_max = ipole

    return (poles[i_max], poles[j_max])


def extract_poles(end_img: SegmentationMask, labels: Labels) -> Dict[int, List[Pole]]:
    """
    Extract poles per cell from ends image.

    Parameters
    ----------
    end_img : 2D array of bool
        'Mask' of the poles in the image.
    labels : 2D array of int
        Cell labels map.

    Returns
    -------
    poles : Dict[int, List[Pole]]
        Dict of cellids to list of poles.  In principle, each list has two
        poles but sometimes the detection fails.

    """
    poles: defaultdict[int, List[Pole]] = defaultdict(list)

    # Append pole locations to poles defaultdict:
    for r, c, lbl in zip(*end_img.nonzero(), labels[end_img]):
        poles[lbl - 1].append(np.array([r, c], dtype=np.int16))

    return dict(poles)


def eucl(p1: npt.NDArray[Any], p2: npt.NDArray[Any]) -> float:
    """
    Euclidean point to point distance.

    Parameters
    ----------
    p1 : 1D array
        Coordinates of first point.
    p2 : 1D array
        Coordinates of second point.

    Returns
    -------
    float
        Euclidean distance between p1 and p2.

    """
    return np.linalg.norm(p1 - p2).astype(float)


def track_poles(features: CellFeatures, prev_old: Pole, prev_new: Pole) -> CellFeatures:
    """
    Track poles of a cell to the previous old and new poles.

    Parameters
    ----------
    features : CellFeatures
        Cell features object with possibly switched poles.
    prev_old : Pole
        Previous old pole of the cell.
    prev_new : Pole
        Previous new pole of the cell.

    Returns
    -------
    features : CellFeatures
        Cell features object with correct poles.

    """
    if (
        eucl(features.old_pole, prev_old) ** 2 + eucl(features.new_pole, prev_new) ** 2
        >= eucl(features.old_pole, prev_new) ** 2
        + eucl(features.new_pole, prev_old) ** 2
    ):
        features.swap_poles()

    return features


def division_poles(
    features1: CellFeatures,
    features2: CellFeatures,
    prev_old: Pole,
    prev_new: Pole,
) -> Tuple[CellFeatures, CellFeatures, bool]:
    """
    Identify which poles belong to the mother and which to the daughter.

    Parameters
    ----------
    features1 : CellFeatures
        Features of one of the 2 cells after division (with possibly swapped poles).
    features2 : CellFeatures
        Features of the other of the 2 cells after division (with possibly swapped poles).
    prev_old : Pole
        Previous old pole of the cell.
    prev_new : Pole
        Previous new pole of the cell.

    Returns
    -------
    mother : CellFeatures
    daughter : CellFeatures
    first_cell_is_mother : bool

    """
    # Find new new poles (2 closest of the poles of the new cells):
    old1_to_2 = min(
        eucl(features1.old_pole, features2.old_pole),
        eucl(features1.old_pole, features2.new_pole),
    )
    new1_to_2 = min(
        eucl(features1.new_pole, features2.old_pole),
        eucl(features1.new_pole, features2.new_pole),
    )
    if old1_to_2 < new1_to_2:
        features1.swap_poles()
    if eucl(features1.new_pole, features2.old_pole) < eucl(
        features1.new_pole, features2.new_pole
    ):
        features2.swap_poles()
    # Now poles are correctly attributed, we will now determine which cell is the mother

    # Track poles closest to old and new pole from previous cell:
    if (
        eucl(features1.old_pole, prev_old) ** 2
        + eucl(features2.old_pole, prev_new) ** 2
        < eucl(features1.old_pole, prev_new) ** 2
        + eucl(features2.old_pole, prev_old) ** 2
    ):
        # cell 1 is mother, cell 2 is daughter
        first_cell_is_mother = True
        mother, daughter = features1, features2

    else:
        # cell 2 is mother, cell 1 is daughter
        first_cell_is_mother = False
        mother, daughter = features2, features1

    return (mother, daughter, first_cell_is_mother)


# %% Lineage


def getTrackingScores(
    labels: Labels,
    outputs: npt.NDArray[np.float32],
    boxes: Optional[List[Tuple[CroppingBox, CroppingBox]]] = None,
) -> npt.NDArray[np.float32]:
    """
    Get overlap scores between input/target cells and tracking outputs.

    Parameters
    ----------
    labels : 2D array of np.uint16
        Labelled image (untracked) of the current frame.
    outputs : 3D array of floats (previous_cells, sizex, sizey)
        Tracking U-Net output (logits).
    boxes : list of dict, optional
        Cropping box and fill box to re-place output prediction masks in the
        original coordinates to index the labels frame.

    Returns
    -------
    scores : 2D array of floats (previous_cells, current_cells)
        Overlap scores matrix between tracking predictions and current
        segmentation mask for each new-old cell.

    """
    # Compile scores:
    scores = np.zeros([outputs.shape[0], labels.max()], dtype=np.float32)
    for o, output in enumerate(outputs):
        # Find pixels with more than 5% probability
        nz_y, nz_x = list((output > logit(0.05)).nonzero())

        # Find cells that are "hit" by those pixels in the labels image:
        if boxes is not None:
            # Clean nz hits outside of fillbox:
            cb, fb = boxes[o][:]
            to_keep_y = (fb.ytl <= nz_y) & (nz_y < fb.ybr)
            to_keep_x = (fb.xtl <= nz_x) & (nz_x < fb.xbr)
            nz_y = nz_y[to_keep_x & to_keep_y]
            nz_x = nz_x[to_keep_x & to_keep_y]

            # Offset hits by cropbox-fillbox:
            nz_y += cb.ytl - fb.ytl
            nz_x += cb.xtl - fb.xtl

        # Compute number of hits per cell:
        cells, counts = np.unique(labels[nz_y, nz_x], return_counts=True)

        # Compile score for these cells:
        for icell, count in zip(cells, counts):
            if icell > 0:
                scores[o, icell - 1] = count

    # Get areas for each cell:
    _, areas = np.unique(labels, return_counts=True)

    return scores / areas[1:]


def getAttributions(scores: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """
    Get attribution matrix from tracking scores.

    Parameters
    ----------
    scores : 2D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : 2D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each old cell can be attributed to 1
        or 2 new cells.

    """
    attrib = np.zeros(scores.shape, dtype=int)

    for n in range(scores.shape[1]):
        # Worst-to-best score of old cells for n-th new cell
        worst_old = np.argsort(scores[:, n])
        # Run through old cells from best to worst:
        for o in range(-1, -scores.shape[0] - 1, -1):
            o_best = worst_old[o]  # o-th best old cell
            s = scores[o_best, n]
            # If score gets too low, stop:
            if s < 0.2:
                break
            # Check if new cell is at least 2nd best for this old cell:
            # Worst-to-best score of new cells for o_best old cell
            worst_new = np.argsort(scores[o_best, :])
            if n in worst_new[-2:]:
                attrib[o_best, n] = 1
                break

    return attrib


class Lineage:
    """Class for cell lineages contained in each ROI."""

    def __init__(self) -> None:
        """
        Initialize object.

        Returns
        -------
        None.

        """
        self.cells: Dict[int, Cell] = {}
        """Dict of cellid to single-cell dictionaries
        Each dictionary will contain single-cell lineage and extracted feature
        properties"""
        self.cellnumbers: List[List[int]] = []
        "List of cells present in FOV at each timepoint"

    def __eq__(self, other) -> bool:
        """Equality function for Lineage."""
        eq_cells = self.cells == other.cells
        eq_cellnumbers = self.cellnumbers == other.cellnumbers
        eq_cellnumbers_empty = all(not x for x in self.cellnumbers) and all(
            not x for x in other.cellnumbers
        )
        return eq_cells and (eq_cellnumbers or eq_cellnumbers_empty)

    def __str__(self) -> str:
        """Display representation of the Lineage."""
        return "\n".join(
            f"cell #{cellid:04}: "
            + "".join(
                " +"[iframe in cell["frames"]]
                for iframe in range(max(cell["frames"]) + 1)
            )
            for cellid, cell in self.cells.items()
        )

    def update(
        self,
        cell: Optional[int],
        frame: int,
        attrib: List[int] = [],
        features: List[CellFeatures] = [],
    ) -> None:
        """
        Attribute cell from previous frame to cell(s) in new frame.

        Parameters
        ----------
        cell : int or None
            Cell index in previous frame to update. If None, the cells in the
            new frame are treated as orphans.
        frame : int
            Frame / timepoint number.
        attrib : list of int, optional
            Cell index(es) in current frame to attribute the previous cell to.
            The default is [].
        features : List[CellFeatures]
            One CellFeature object for each cell id in attrib.
            The default is [].

        Returns
        -------
        None.

        """
        # Check cell numbers list:
        if len(self.cellnumbers) <= frame:
            self.cellnumbers += [
                [] for _ in range(frame + 1 - len(self.cellnumbers))
            ]  # Initialize
        if len(attrib) > 0 and len(self.cellnumbers[frame]) <= max(attrib):
            self.cellnumbers[frame] += [
                -1 for _ in range(max(attrib) + 1 - len(self.cellnumbers[frame]))
            ]  # Initialize

        # If no attribution: (new/orphan cell)
        if cell is None and len(attrib) > 0:
            assert len(attrib) == 1
            # Arbitrarily set "higher" pole as old pole:
            if features[0].new_pole[0] <= features[0].old_pole[0]:
                features[0].swap_poles()
            cellnum = self.createcell(frame, features=features[0])
            self.cellnumbers[frame][attrib[0]] = cellnum

        # Simple tracking event:
        elif len(attrib) == 1:
            cellnum = self.cellnumbers[frame - 1][cell]  # Get old cell number
            # Update cellnumbers list:
            self.cellnumbers[frame][attrib[0]] = cellnum
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(cellnum, frame - 1, "old_pole")
            prev_new = self.getvalue(cellnum, frame - 1, "new_pole")
            assert isinstance(prev_old, np.ndarray)  # for mypy
            assert isinstance(prev_new, np.ndarray)  # for mypy
            # Find which pole is new vs old:
            feats = track_poles(features[0], prev_old, prev_new)
            # Update:
            self.updatecell(cellnum, frame, features=feats)

        # Division event:
        elif len(attrib) == 2:
            mothernum = self.cellnumbers[frame - 1][cell]  # Get old cell number
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(mothernum, frame - 1, "old_pole")
            prev_new = self.getvalue(mothernum, frame - 1, "new_pole")
            assert isinstance(prev_old, np.ndarray)  # for mypy
            assert isinstance(prev_new, np.ndarray)  # for mypy
            # Identify mother and daughter poles:
            mother, daughter, first_cell_is_mother = division_poles(
                features[0], features[1], prev_old, prev_new
            )
            # Create daughter:
            daughternum = self.createcell(
                frame,
                features=daughter,
                mother=mothernum,
            )
            # Update mother:
            self.updatecell(
                mothernum,
                frame,
                features=mother,
                daughter=daughternum,
            )
            # Update cellnumbers list:
            if first_cell_is_mother:
                self.cellnumbers[frame][attrib[0]] = mothernum
                self.cellnumbers[frame][attrib[1]] = daughternum
            else:
                self.cellnumbers[frame][attrib[0]] = daughternum
                self.cellnumbers[frame][attrib[1]] = mothernum

    def createcell(
        self,
        frame: int,
        features: CellFeatures,
        mother: Optional[int] = None,
    ) -> int:
        """
        Create cell to append to lineage list.

        Parameters
        ----------
        frame : int
            Frame that the cell first appears (1-based indexing).
        features : CellFeatures
            CellFeatures object to instantiate from.
        mother : int or None, optional
            Number of the mother cell in the lineage (1-based indexing).
            The default is None. (ie unknown mother)

        Returns
        -------
        int
            new cell number

        """
        new_cell = Cell(
            {
                "id": len(self.cells),
                "mother": mother,
                "frames": [frame],
                "daughters": [None],
                "new_pole": [features.new_pole],
                "old_pole": [features.old_pole],
                "length": [features.length],
                "width": [features.width],
                "area": [features.area],
                "perimeter": [features.perimeter],
                "fluo": [features.fluo],
                "edges": [features.edges],
                "growthrate_length": [features.growthrate_length],
                "growthrate_area": [features.growthrate_area],
            }
        )
        self.cells[new_cell["id"]] = new_cell

        return new_cell["id"]

    def updatecell(
        self,
        cell: int,
        frame: int,
        features: CellFeatures,
        daughter: Optional[int] = None,
    ) -> None:
        """
        Update cell lineage values.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        features : CellFeatures
            CellFeatures object to update from.
        daughter : int or None, optional
            Daughter cell number if division just happened.
            The default is None.

        Returns
        -------
        None.

        """
        self.cells[cell]["frames"] += [frame]
        self.cells[cell]["daughters"] += [daughter]
        self.cells[cell]["old_pole"] += [features.old_pole]
        self.cells[cell]["new_pole"] += [features.new_pole]
        self.cells[cell]["length"] += [features.length]
        self.cells[cell]["width"] += [features.width]
        self.cells[cell]["area"] += [features.area]
        self.cells[cell]["perimeter"] += [features.perimeter]
        self.cells[cell]["fluo"] += [features.fluo]
        self.cells[cell]["edges"] += [features.edges]
        self.cells[cell]["growthrate_length"] += [features.growthrate_length]
        self.cells[cell]["growthrate_area"] += [features.growthrate_area]

    def setvalue(
        self, cell: int, frame: int, feature: str, value: Union[int, float, str]
    ) -> None:
        """
        Set feature value for specific cell and frame/timepoint.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to set value for.
        value : int, float, str
            Value to assign.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        None.

        """
        try:
            i = self.cells[cell]["frames"].index(frame)
        except ValueError as err:
            raise ValueError(f"Cell {cell} is not present in frame {frame}") from err

        # Get cell dict
        cell_dict = self.cells[cell]

        # If feature doesn't exist yet, create list:
        if feature not in cell_dict:
            cell_dict[feature] = [None for _ in range(len(cell_dict["frames"]))]

        # Add value:
        cell_dict[feature][i] = value

    def getpastvalue(self, cell: int, frame: int, feature: str, df: int = 1) -> float:
        """
        Get feature value for a frame in the past, ignoring cell divisions.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Current frame number.
        feature : str
            Feature to get value for. Only works for numerical scalar features.
        df : int
            Number of frames in the past where we want to get the value.

        Returns
        -------
        float
            The value in the past, normalized by ratio of the value at current
            timepoint for initial cell by sum of value all progeny of the past
            cell (if any sister/cousin cells)

        """
        # This function works in two steps. First, it determines the ancestor
        # cell of cell `cell` at frame `frame - df`. Then, it determines what
        # percentage of the current progeny of the ancestor, the cell `cell`
        # represents.

        # This loop computes the ancestor cell of cell `cell` and puts its
        # number into `ancestor`.
        value_ancestor = None
        ancestor = cell
        while value_ancestor is None:
            try:
                # Did the current cell exist at the specified frame? If so, the search is over.
                value_ancestor = self.getvalue(
                    ancestor, frame=frame - df, feature=feature
                )
            except ValueError:
                # The cell did not exist at the specified frame, so we try with its mother.
                ancestor = self.cells[ancestor]["mother"]
                # If the cell has no mother, we cannot continue.
                if ancestor is None:
                    return None

        # At this point, `ancestor` and `value_ancestor` are correctly filled.
        # Now we will collect the value of all the progeny of this cell, at the
        # current frame.
        value_now = self.getfuturevalue(ancestor, frame - df, feature, df=df)
        # If we cannot do that, it means that we lost track of a children. We exit.
        if value_now is None or value_now == 0.0:
            return None

        # At this point, we have everything needed. We can return the value of
        # the ancestor, weighted by the ratio of the current cell compared to
        # its sisters/cousins/etc.
        return self.getvalue(cell, frame, feature) * value_ancestor / value_now

    def getfuturevalue(self, cell: int, frame: int, feature: str, df: int = 1) -> float:
        """
        Get feature value for a frame in the future, ignoring cell divisions.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Current frame number.
        feature : str
            Feature to get value for. Only works for numerical scalar features.
        df : int
            Number of frames in the future where we want to get the value.

        Returns
        -------
        value: float
            Sum of value for all progeny of initial cell in the future.
        """
        # This function makes progress recursively.
        # Here we handle the base case where the frame requested is the current one.
        if df == 0:
            return self.getvalue(cell, frame, feature)

        # Here we have df > 0, so we make progress to the next frame,
        # considering the current branch, and the potential branch of a
        # daughter cell if one exists at next frame.

        try:
            # Given our cell numbering, the mother cell has to exist at next
            # frame, unless it gets out of frame or there is a tracking error.
            # In any of these cases, we have to exit.
            value = self.getfuturevalue(
                cell, frame=frame + 1, feature=feature, df=df - 1
            )
        except ValueError:
            return None

        # If the cell has a daughter at next frame, we consider this branch as well.
        idaughter = self.getvalue(cell, frame + 1, "daughters")
        if idaughter is not None:
            assert isinstance(idaughter, int)
            value += self.getfuturevalue(
                idaughter, frame=frame + 1, feature=feature, df=df - 1
            )
        return value

    def getvalue(self, cell: int, frame: int, feature: str) -> Union[int, float, str]:
        """
        Get feature value for specific timepoint/frame.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to get value for.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        int, float, str
            Value for feature at frame.

        """
        try:
            i = self.cells[cell]["frames"].index(frame)
        except ValueError as err:
            raise ValueError(f"Cell {cell} is not present in frame {frame}") from err

        return self.cells[cell][feature][i]

    def growthrate(self, cell: int, frame: int, feature: str) -> Optional[float]:
        """
        Compute exponential growth rate of a given feature for the given cell at the given frame.

        Compute the central derivative with the previous and the next frame if
        possible, otherwise a one-sided derivative.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to get value for.

        Returns
        -------
        Optional[float]
            Growth rate of the feature at frame.
        """
        value_fm1 = self.getpastvalue(cell, frame, feature)
        value_f = self.getvalue(cell, frame, feature)
        assert isinstance(value_f, int) or isinstance(value_f, float)
        value_fp1 = self.getfuturevalue(cell, frame, feature)
        if (
            value_fm1 is not None
            and value_fp1 is not None
            and value_fm1 > 0.0
            and value_fp1 > 0.0
        ):
            return np.log(value_fp1 / value_fm1) / 2.0
        if value_fm1 is not None and value_fm1 > 0.0 and value_f > 0.0:
            return np.log(value_f / value_fm1)
        if value_fp1 is not None and value_f > 0.0 and value_fp1 > 0.0:
            return np.log(value_fp1 / value_f)
        return None


# %% Image files


def getxppathdialog(ask_folder: bool = False) -> str:
    """
    Pop up window to select experiment file or folder.

    Parameters
    ----------
    ask_folder : bool, optional
        Folder selection window will pop up instead of file selection.
        The default is False.

    Returns
    -------
    file_path : str
        Path to experiment file or folder.

    """
    root = tk.Tk()
    # root.withdraw() # For some reason this doesn't work with askdirectory?
    if ask_folder:
        file_path = filedialog.askdirectory(
            title="Please select experiment folder(s)", mustexist=True
        )

    else:
        file_path = filedialog.askopenfilename(
            title="Please select experiment files(s)"
        )

    root.destroy()

    return file_path


class xpreader:
    """Class to read experiment files from bioformats-compatible files or from file sequences in folders."""

    def __init__(
        self,
        filename: Optional[Union[Path, str]] = None,
        channelnames: Optional[Tuple[str]] = None,
        use_bioformats: bool = False,
        prototype: Optional[str] = None,
        fileorder: str = "pct",
        filenamesindexing: int = 1,
    ) -> None:
        """
        Initialize experiment reader.

        Parameters
        ----------
        filename : Path or str, optional
            Path to experiment file or directory. If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If None, an interactive selection dialog
            will be used. If no prototype is provided, the filenames are
            expected to be of the following C-style format:
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
            - Pos01_Cha3_Fra0005.tif
            - p3c1t034.tif
            - xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing : int
            Selects between 0-based or 1-based indexing in the filename. If
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1

        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        """
        # Set default parameters
        self.filename = None if not filename else Path(filename)
        "File or folder name for the experiment"
        self.use_bioformats = use_bioformats
        "Flag to use the Bio-Formats library to read file"
        self.fileorder = fileorder
        "File ordering (if image files sequence)"
        self.filenamesindexing = filenamesindexing
        "File indexing start (if image files sequence)"
        self.prototype = prototype
        "File name prototype (if image files sequence)"

        # Init base parameters:
        self.filetype = None
        "Type / extension of `filehandle`"
        self.filehandle: Any
        "Handle to file reader or base directory"
        self.positions: int
        "Number of positions in experiment"
        self.channels: int
        "Number of imaging channels"
        self.x: int
        "Size of images along X axis"
        self.y: int
        "Size of images along Y axis"
        self.dtype: str
        "Datatype of images"

        if filename is None:
            # Interactive selection and pop up window:
            self._interactive_init()

        self.filename = Path(self.filename)

        self.channelnames = channelnames
        "Names of the imaging channels (optional)"

        file_extension = self.filename.suffix

        if self.use_bioformats:
            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(str(self.filename))
            md = bioformats.OMEXML(
                bioformats.get_omexml_metadata(path=str(self.filename))
            )
            self.positions = md.get_image_count()
            self.timepoints = md.image(
                0
            ).Pixels.SizeT  # Here I'm going to assume all series have the same number of timepoints
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=True, c=0).dtype

        elif self.filename.is_dir():
            # Experiment is stored as individual image TIFF files in a folder
            self.filetype = "dir"
            self.filehandle = self.filename
            # If filename prototype is not provided, guess it from the first file:
            if self.prototype is None:
                imgfiles = [
                    x
                    for x in self.filename.iterdir()
                    if x.suffix.lower() in (".tif", ".tiff")
                ]
                # Here we assume all images in the folder follow the same naming convention:
                # Get digits sequences in first filename
                numstrs = re.findall(r"\d+", imgfiles[0].name)
                # Get character sequences in first filename
                charstrs = re.findall(r"\D+", imgfiles[0].name)
                if len(numstrs) != 3 or len(charstrs) != 4:
                    raise ValueError(
                        "Filename formatting error. See documentation for image sequence formatting"
                    )
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype = (
                    f"{charstrs[0]}%0{len(numstrs[0])}d"
                    f"{charstrs[1]}%0{len(numstrs[1])}d"
                    f"{charstrs[2]}%0{len(numstrs[2])}d"
                    f"{charstrs[3]}"
                )
            # Get experiment settings by testing if relevant files exist:
            # Get number of positions:
            if "p" in self.fileorder:
                self.positions = 0
                while self.getfilenamefromprototype(self.positions, 0, 0).exists():
                    self.positions += 1
            else:
                self.positions = 1
            # Get number of channels:
            if "c" in self.fileorder:
                self.channels = 0
                while self.getfilenamefromprototype(0, self.channels, 0).exists():
                    self.channels += 1
            else:
                self.channels = 1
            # Get number of frames/timepoints:
            if "t" in self.fileorder:
                self.timepoints = 0
                while self.getfilenamefromprototype(0, 0, self.timepoints).exists():
                    self.timepoints += 1
            else:
                self.timepoints = 1  # I guess this shouldn't really happen
            # Get image specs:
            # Load first image, get image data from it
            img = read_image(self.getfilenamefromprototype(0, 0, 0))
            self.x = img.shape[1]
            self.y = img.shape[0]
            self.dtype = img.dtype

        elif file_extension.lower() == ".tif" or file_extension.lower() == ".tiff":
            # Works with single-series tif & multi-series ome.tif
            self.filetype = "tif"

            self.filehandle = TiffFile(str(self.filename))
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[
                0
            ]  # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find("T")]
            self.channels = s.shape[s.axes.find("C")]
            self.x = s.shape[s.axes.find("X")]
            self.y = s.shape[s.axes.find("Y")]
            self.dtype = s.pages[0].asarray().dtype

    def _interactive_init(self) -> None:
        """
        Interactive initialization routine.

        Raises
        ------
        ValueError
            If a non-valid experiment type was passed.

        Returns
        -------
        None.

        """
        # Get xp settings:
        print(
            (
                "Experiment type?\n"
                "1 - Bio-Formats compatible (.nd2, .oib, .czi, .ome.tif...)\n"
                "2 - bioformats2sequence (folder)\n"
                "3 - micromanager (folder)\n"
                "4 - high-throughput (folder)\n"
                "0 - other (folder)\n"
                "Enter a number: "
            ),
            end="",
        )
        answer = int(input())
        print()

        # If bioformats file(s):
        if answer is None or answer == 1:
            print("Please select experiment file(s)...")
            self.filename = getxppathdialog(ask_folder=False)
            self.use_bioformats = True
            self.prototype = None
            self.filenamesindexing = 1
            self.fileorder = "pct"

        # If folder:
        else:
            print("Please select experiment folder...")
            self.filename = getxppathdialog(ask_folder=True)
            self.use_bioformats = False
            if answer is None or answer == 2:
                self.prototype = None
                self.fileorder = "pct"
                self.filenamesindexing = 1
            elif answer == 3:
                self.prototype = (
                    "Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif"
                )
                self.fileorder = "pcpt"
                self.filenamesindexing = 0
            elif answer == 4:
                self.prototype = "chan%02d_img/Position%06d_Frame%06d.tif"
                self.fileorder = "cpt"
                self.filenamesindexing = 1
            elif answer == 0:
                print("Enter files prototype: ", end="")
                self.prototype = input()
                print()
                print("Enter files order: ", end="")
                self.fileorder = input()
                print()
                print("Enter files indexing: ", end="")
                self.filenamesindexing = int(input())
                print()
            else:
                raise ValueError("Invalid experiment type")
            print()

    def close(self) -> None:
        """Close the bioformats or tif reader."""
        if self.use_bioformats:
            self.filehandle.close()
            javabridge.kill_vm()
        elif self.filetype == "tif":  # Nothing to do if sequence directory
            self.filehandle.close()

    def getfilenamefromprototype(self, position: int, channel: int, frame: int) -> Path:
        """
        Generate full filename for specific frame based on file path, prototype, fileorder, and filenamesindexing.

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        """
        filenumbers = tuple(
            {"p": position, "c": channel, "t": frame}[c] + self.filenamesindexing
            for c in self.fileorder
        )
        assert self.prototype is not None  # FIXME: why is it not?
        return self.filehandle / (self.prototype % filenumbers)

    # TODO: typecheck this function
    def getframes(
        self,
        position: int,
        channels: Optional[Union[int, Sequence[int]]] = None,
        nbframes: Optional[int] = None,
        squeeze_dimensions: bool = True,
        rotate: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get frames from experiment.

        Parameters
        ----------
        position : int
            The position index for which the frames are requested.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer
            or a tuple/list will be returned. If the channel names have been
            defined, the channel(s) can be passed as a string or tuple/list of
            strings. If an empty list is passed, None is returned. If None is
            passed, all channels are returned.
            The default is None.
        nbframes : int or None, optional
            Number of frames returned, starting from the first frame.  If None
            is passed, all frames are returned.
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 4-dimensional, with the
            dimensions order being: Time, Channel, Y, X

        """
        # Handle options:
        if channels is None:  # No channel specified: load all
            channels = list(range(self.channels))
        elif isinstance(channels, str):  # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError("Set channel names first")
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels + " is not a valid channel name.")
        elif isinstance(channels, (list, tuple)):
            # If list of ints, nothing to do
            if channels and all(isinstance(c, str) for c in channels):
                # refer to channels by list/tuple of names:
                if self.channelnames is None:
                    raise ValueError("Set channel names first")
                channels_list = []
                for channel in channels:
                    if channel in self.channelnames:
                        channels_list.append(self.channelnames.index(channel))
                    else:
                        raise ValueError(channel + " is not a valid channel name.")
                channels = channels_list
        elif not isinstance(channels, (list, tuple)):
            assert isinstance(channels, int)
            channels = [channels]

        if nbframes is None:
            frames = list(range(self.timepoints))
        else:
            frames = list(range(nbframes))

        # Allocate memory:
        output = np.empty(
            [len(frames), len(channels), self.y, self.x], dtype=np.float32
        )

        # Load images:
        for c, cha in enumerate(channels):
            for f, fra in enumerate(frames):
                # Read frame:
                if self.use_bioformats:
                    frame = self.filehandle.read(
                        series=position, c=cha, t=fra, rescale=True
                    )
                elif self.filetype == "dir":
                    frame = read_image(
                        self.getfilenamefromprototype(position, cha, fra)
                    )
                elif self.filetype == "tif":
                    frame = (
                        self.filehandle.series[position]
                        .pages[fra * self.channels + cha]
                        .asarray()
                    )

                # Optionally rotate
                if rotate is not None:
                    frame = imrotate(frame, rotate)
                # Add to output array:
                output[f, c, :, :] = frame

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output


# %% Saving & Loading results


def getrandomcolors(
    cellids: Sequence[int], seed: int = 0
) -> Dict[int, Tuple[float, float, float]]:
    """
    Generate list of random hsv colors.

    Parameters
    ----------
    cellids : Iterable (for example list, tuple or iterator)
        Cellids which should be assigned random colors.
    seed : int, optional
        Random seed used to shuffle colors.

    Returns
    -------
    colors : Dict[int, Tuple[float, float, float]]
        Dictionary of cellids to colors (RGB values in [0,1]).

    """
    # XXX: Is this necessary?
    if len(cellids) == 0:
        return {}

    # Get colors
    gradient = np.linspace(0, 256, len(cellids), endpoint=False, dtype=np.uint8)
    colors = cv2.applyColorMap(gradient, cv2.COLORMAP_HSV).squeeze(axis=1)
    colors = colors.astype(np.float64) / 255

    # Shuffle colors
    rng = npr.default_rng(seed)
    rng.shuffle(colors)

    return {cellid: tuple(color) for cellid, color in zip(cellids, colors)}


def vidwrite(
    images: np.ndarray, filename: Union[str, Path], crf: int = 20, verbose: int = 1
) -> None:
    """
    Write images stack to video file with h264 compression.

    Parameters
    ----------
    images : 4D numpy array
        Stack of RGB images to write to video file.
    filename : str or Path
        File name to write video to. (Overwritten if exists)
    crf : int, optional
        Compression rate. 'Sane' values are 17-28. See
        https://trac.ffmpeg.org/wiki/Encode/H.264
        The default is 20.
    verbose : int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    None.

    """
    # Initialize ffmpeg parameters:
    height, width, _ = images[0].shape
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    quiet = [] if verbose else ["-loglevel", "error", "-hide_banner"]
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            r=7,
        )
        .output(
            str(filename),
            pix_fmt="yuv420p",
            vcodec="libx264",
            crf=crf,
            preset="veryslow",
        )
        .global_args(*quiet)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write frames:
    for frame in images:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()


# %% Feature extraction


def roi_features(
    labels_frame: npt.NDArray[np.uint16],
    fluo_frames: np.ndarray,
    roi_box: CroppingBox,
) -> Dict[int, CellFeatures]:
    """
    Extract single-cell morphological and fluorescence features.

    Parameters
    ----------
    labels_frame : 2D array of uint16
        Labels image of numbered cell regions.
    fluo_frames : 3D array
        Array of fluo frames to extract fluorescence from. Dimensions are
        (size_x, size_y, fluo_channels). The number of fluo_channels must match
        the number of fluo features to extract.
    roi_box : CroppingBox
        Cropping box to apply shift when looking at full-frame fluo images.

    Returns
    -------
    cell_features : Dict[int, CellFeatures]
        Dictionary that associates cellids and CellFeatures.
    """
    # Cell numbers and contours:
    cellids, contours = getcellsinframe(labels_frame, return_contours=True)

    cell_features = {}
    # Loop through cells in image, extract single-cell features:
    for cellid, contour in zip(cellids, contours):
        mask = labels_frame == cellid + 1

        fluo_values = cell_fluo(fluo_frames, mask, roi_box)

        width, length = cell_width_length(contour)

        cell_features[cellid] = CellFeatures(
            new_pole=None,
            old_pole=None,
            edges=image_edges(contour, mask),
            width=width,
            length=length,
            area=cell_area(contour),
            perimeter=cell_perimeter(contour),
            fluo=fluo_values,
        )

    return cell_features


def image_edges(contour: Contour, image: npt.NDArray[Any]) -> str:
    """
    Identify if cell touches image borders.

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.
    image : 2D numpy array
        Image where the cell is present.

    Returns
    -------
    edge_str : str
        String describing edges touched by the cell. Can be a combination of
        the following strs: '-x', '+x', '-y', '+y'. Empty otherwise.

    """
    edge_str = ""
    if any(contour[:, 0, 0] == 0):
        edge_str += "-x"
    if any(contour[:, 0, 0] == image.shape[1] - 1):
        edge_str += "+x"
    if any(contour[:, 0, 1] == 0):
        edge_str += "-y"
    if any(contour[:, 0, 1] == image.shape[0] - 1):
        edge_str += "+y"

    return edge_str


def cell_width_length(contour: Contour) -> Tuple[float, float]:
    """
    Measure width and length of single cell.

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    width : float
        Cell width.
    length : float
        cell length.

    """
    # Get rotated bounding box:
    _, size, _ = cv2.minAreaRect(contour)

    width = min(size)
    length = max(size)

    return width, length


def cell_area(contour: Contour) -> float:
    """
    Area of a single cell.

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    area : float
        Cell area

    """
    area = cv2.contourArea(contour)

    return area


def cell_perimeter(contour: Contour) -> float:
    """
    Get single cell perimeter.

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    perimeter : int
        Cell perimeter

    """
    perimeter = cv2.arcLength(contour, closed=True)

    return perimeter


def cell_fluo(
    fluo_frames: np.ndarray,
    mask: SegmentationMask,
    box: CroppingBox,
) -> List[float]:
    """
    Extract mean fluorescence level from mask.

    Parameters
    ----------
    fluo_frames : 3D array
        Fluorescent images to extract fluo from. Dimensions are
        (channels, size_y, size_x).
    mask : 2D numpy array of bool
        Mask of the region to extract (typically a single cell).
    box : CroppingBox, optional
        Cropping box to apply shift when looking at full-frame fluo images.
        The default is None.

    Returns
    -------
    fluo_values :  list of floats
        Mean value per cell for each fluo frame.

    """
    return [np.sum(mask * box.crop(fluo)) / mask.sum() for fluo in fluo_frames]


# %% Misc


def read_image(filename: Path) -> np.ndarray:
    """
    Return the image as a numpy array of np.float32 rescaled between 0 and 1.

    Parameters
    ----------
    filename : Path
        Filename of the image to read.

    Returns
    -------
    image : np.ndarray of np.float32
    """
    img = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"File not found or format not understood: {filename}.")
    for depth, dtype in [(8, np.uint8), (16, np.uint16), (32, np.uint32)]:
        if img.dtype == dtype:
            return img.astype(np.float32) / (2**depth - 1)
    raise ValueError(f"Image depth not understood: {img.dtype}.")


def findfirst(mylist: List) -> Optional[int]:
    """
    Find first non-zero element of list.

    Parameters
    ----------
    mylist : list
        List of elements to scan through.

    Returns
    -------
    int or None
        Index of first non-zero element, if it exists.

    """
    return next((i for i, x in enumerate(mylist) if x > 0), None)


def listfiles(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    """
    Return a sorted list of filenames in `directory` that have one of the specified extensions.

    Parameters
    ----------
    directory : Path
        Directory to iterate.
    suffixes : Iterable[str]
        List, tuple, set, etc. of allowed extensions.

    Returns
    -------
    [Path]
        List of matching filenames.
    """
    return sorted(p for p in directory.iterdir() if p.suffix in suffixes)


def tensorboard_callback() -> TensorBoard:
    """
    Return a callback for TensorBoard logging.

    Returns
    -------
    TensorBoard
        TensorBoard callback.
    """
    try:
        branch = (
            subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"])
            .rstrip()
            .decode()
        )
    except FileNotFoundError:
        branch = ""
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{date}_{branch}"
    callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    return callback
