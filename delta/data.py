"""
Module for data manipulations and input/output operations.

@author: jblugagne
"""

import copy
import importlib
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import cv2
import numpy as np
import numpy.random as npr
import numpy.typing as npt
import skimage.morphology as morph
import skimage.transform as trans
import tensorflow as tf
from scipy import interpolate
from skimage import io
from skimage.measure import label

# Try to import elastic deformations, issue warning if not found:
if importlib.util.find_spec("elasticdeform") is None:
    warnings.warn("Could not load elastic deformations module.")
else:
    import elasticdeform

from delta import utilities as utils

#%% UTILITIES:


def binarizerange(array: utils.Image, threshold: Optional[float] = None) -> utils.Image:
    """
    This function will binarize a numpy array by thresholding it.

    Parameters
    ----------
    array : 2D numpy array of floating-point numbers
        Input array/image.
    threshold : float, optional
        Binarization threshold.
        By default, the threshold will be the middle of the dynamic range.

    Returns
    -------
    newi : 2D numpy array of uint8
           Binarized image.

    """
    if threshold is None:
        threshold = (np.amin(array) + np.amax(array)) / 2
    return (array > threshold).astype(np.uint8)


def readreshape(
    filename: Path,
    target_size: Tuple[int, int] = (256, 32),
    binarize: bool = False,
    order: int = 1,
    method: str = "resize",
) -> npt.NDArray[Any]:
    """
    Read image from disk, format it and return it as an array of
    floating-point numbers between 0 and 1.

    Parameters
    ----------
    filename : Path
        Path to file. Only PNG, JPG or single-page TIFF files accepted
    target_size : tupe of int or None, optional
        Size to reshape the image.
        The default is (256, 32).
    binarize : bool, optional
        Use the binarizerange() function on the image.
        The default is False.
    order : int, optional
        interpolation order (see skimage.transform.warp doc).
        0 is nearest neighbor
        1 is bilinear
        The default is 1.
    method : "resize" or "pad"
        What to do if the image is smaller than the target size.
        In the "resize" mode, larger images will be resized too.
        The default is "resize".

    Raises
    ------
    ValueError
        Raised if image file is not a PNG, JPEG, or TIFF file.

    Returns
    -------
    i : numpy 2d array of floats
        Loaded array.

    """
    i = utils.read_image(filename)
    assert i.ndim == 2
    if method == "pad":
        # For DeLTA 2D, black space is added if img is smaller than target_size
        fill_shape = np.maximum(np.array(target_size), np.array(i.shape))
        img = np.zeros(fill_shape)
        img[: i.shape[0], : i.shape[1]] = i
    elif method == "resize":
        # For DeLTA mothermachine, all images are resized in 256x32
        img = trans.resize(i, target_size, anti_aliasing=True, order=order)
    else:
        raise ValueError(f"Resizing method not understood: {method}.")

    if binarize:
        return binarizerange(img).astype(img.dtype)

    return img


def postprocess(
    images: npt.NDArray[np.float32],
    square_size: int = 5,
    min_size: Optional[float] = None,
    crop: bool = False,
) -> npt.NDArray[np.uint8]:
    """
    A generic binary image cleaning function based on mathematical morphology.

    Parameters
    ----------
    images : 2D or 3D numpy array of floating-point numbers between 0 and 1
        Input image or images stacked along axis=0.
    square_size : int, optional
        Size of the square structuring element to use for morphological opening
        The default is 5.
    min_size : float or None, optional
        Remove objects smaller than this minimum area value. If None, the
        operation is not performed.
        The default is None.

    Returns
    -------
    images : 2D or 3D numpy array of uint8 (0 or 1)
        Cleaned, binarized images. Note that the dimensions are squeezed before
        return (see numpy.squeeze doc)

    """
    # Expand dims if 2D:
    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)

    outputs = []
    for image in images:
        image = binarizerange(image, threshold=0)
        if not crop:
            kernel = np.ones((square_size, square_size), dtype=np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if min_size is not None:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(image)
            threshold = (stats[:, cv2.CC_STAT_AREA] >= min_size).astype(np.uint8)
            threshold[0] = 0
            image = threshold[labels]
        outputs.append(image)

    return np.squeeze(outputs)


#%% DATA AUGMENTATION
def data_augmentation(
    images_input: List[npt.NDArray[Any]],
    aug_par: Dict[str, Any],
    order: Union[int, List[int], Tuple[int, ...]] = 0,
    rng: Optional[npr.Generator] = None,
) -> List[npt.NDArray[Any]]:
    """
    Data augmentation function.

    Parameters
    ----------
    images_input : list of 2D numpy arrays
        Images to apply augmentation operations to.
    aug_par : dict
        Augmentation operations parameters. Accepted key-value pairs:
            illumination_voodoo: bool.
                Whether to apply the illumination voodoo operation.
            histogram_voodoo: bool.
                Whether to apply the histogram voodoo operation.
            elastic_deformation: dict.
                If key exists, the elastic deformation operation is applied.
                The parameters are given as key-value pairs. sigma values are
                given under the sigma key, deformation points are given under
                the points key. See elasticdeform doc.
            gaussian_noise: float.
                Apply gaussian noise to the image. The sigma value of the
                gaussian noise is uniformly sampled between 0 and +gaussian_noise.
            gaussain_blur: float.
                Apply gaussian blur to the image. The sigma value is the
                standard deviation of the kernel in the x and y direction.
            horizontal_flip: bool.
                Whether to flip the images horizontally. Input images have a
                50% chance of being flipped
            vertical_flip: bool.
                Whether to flip the images vertically. Input images have a 50%
                chance of being flipped
            rotations90d: bool.
                Whether to randomly rotate the images in 90° increments.
                Each 90° rotation has a 25% chance of happening
            rotation: int or float.
                Range of random rotation to apply. The angle is uniformly
                sampled in the range [-rotation, +rotation]
            zoom: float.
                Range of random "zoom" to apply. The image is randomly zoomed
                by a factor that is sampled from an exponential distribution
                with a lamba of 3/zoom. The random factor is clipped at +zoom.
            shiftX: int/float.
                The range of random shifts to apply along X. A uniformly
                sampled shift between [-shiftX, +shiftX] is applied
            shiftY: int/float.
                The range of random shifts to apply along Y. A uniformly
                sampled shift between [-shiftY, +shiftY] is applied

            Note that the same operations are applied to all inputs except for
            the timeshift ones.
    order : int or list/tuple of ints, optional
        Interpolation order to use for each image in the input stack. If order
        is a scalar, the same order is applied to all images. If a list of
        orders is provided, each image in the stack will have its own operation
        order. See skimage.transform.wrap doc.
        Note that the histogram voodoo operation is only applied to images with
        a non-zero order.
        The default is 0.

    Returns
    -------
    output : list of 2D numpy arrays
        Augmented images array.

    """
    if rng is None:
        rng = npr.default_rng()

    # processing inputs / initializing variables::
    output = list(images_input)
    orderlist = [order] * len(images_input) if isinstance(order, int) else list(order)

    # Apply augmentation operations:

    if aug_par.get("illumination_voodoo"):
        output = [illumination_voodoo(item, rng) if order > 0 else item
                  for order, item in zip(orderlist, output)]

    if aug_par.get("histogram_voodoo"):
        output = [histogram_voodoo(item, rng) if order > 0 else item
                  for order, item in zip(orderlist, output)]

    if aug_par.get("gaussian_noise"):
        output = [np.clip(rng.normal(item, aug_par["gaussian_noise"]), 0, 1) if order > 0 else item
                  for order, item in zip(orderlist, output)]

    if aug_par.get("gaussian_blur"):
        output = [cv2.GaussianBlur(item, (5, 5), aug_par["gaussian_blur"]) if order > 0 else item
                  for order, item in zip(orderlist, output)]

    if aug_par.get("elastic_deformation"):
        output = elasticdeform.deform_random_grid(
            output,
            sigma=aug_par["elastic_deformation"]["sigma"],
            points=aug_par["elastic_deformation"]["points"],
            # Using bicubic interpolation instead of bilinear here
            order=[i * 3 for i in orderlist],
            mode="nearest",
            axis=(0, 1),
            prefilter=False,
        )

    if aug_par.get("horizontal_flip"):
        if rng.choice(2):
            output = [np.fliplr(item) for item in output]

    if aug_par.get("vertical_flip"):
        if rng.choice(2):
            output = [np.flipud(item) for item in output]

    if aug_par.get("rotations_90d"):
        output = [np.rot90(item, k=rng.choice(4)) for item in output]

    if aug_par.get("rotation"):
        rot = rng.uniform(-aug_par["rotation"], aug_par["rotation"])
        output = [trans.rotate(item, rot, mode="edge", order=order) if order > 0 else item
                  for order, item in zip(orderlist, output)]

    # Zoom and shift operations are processed together:
    if aug_par.get("zoom"):
        # I want most of them to not be too zoomed
        zoom = rng.exponential(aug_par["zoom"] / 3.0)
        zoom = min(zoom, aug_par["zoom"])
    else:
        zoom = 0

    shift_x = rng.uniform(-1.0, 1.0) * aug_par.get("shiftX", 0.0)
    shift_y = rng.uniform(-1.0, 1.0) * aug_par.get("shiftY", 0.0)

    # Apply zoom & shifts:
    if any([zoom, shift_x, shift_y]):
        output = [zoomshift(item, zoom + 1, shift_x, shift_y, order=order)
                  for order, item in zip(orderlist, output)]

    return output


def zoomshift(
    image: utils.Image, zoomlevel: float, shiftX: float, shiftY: float, order: int = 0
) -> utils.Image:
    """
    This function zooms and shifts images.

    Parameters
    ----------
    I : 2D numpy array
        input image.
    zoomlevel : float
        Additional zoom to apply to the image.
    shiftX : float
        X-axis shift to apply to the image, in pixels.
    shiftY : float
        Y-axis shift to apply to the image, in pixels.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    I : 2D numpy array
        Zoomed and shifted image of same size as input.

    """

    # Center of rotation and zoom
    center = (np.array(image.shape[1::-1]) - 1) / 2
    # Affine transform matrix
    matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=zoomlevel)
    # Add shift
    matrix[:, 2] -= [shiftY*image.shape[1]*zoomlevel, shiftX*image.shape[0]*zoomlevel]

    flags = [cv2.INTER_NEAREST, cv2.INTER_LINEAR][order]
    new_image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return new_image


def histogram_voodoo(
    image: utils.Image, rng: npr.Generator, num_control_points: int = 3
) -> utils.Image:
    """
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination.

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    rng : npr.Generator
        Pseudo-random number generator.
    num_control_points : int, optional
        Number of inflection points to use on the histogram conversion curve.
        The default is 3.

    Returns
    -------
    2D numpy array
        Modified image.

    """
    control_points = np.linspace(0, 1, num=num_control_points + 2)
    sorted_points = np.array(control_points)
    random_points = rng.uniform(low=0.1, high=0.9, size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)

    return mapping(image)


def illumination_voodoo(
    image: utils.Image, rng: npr.Generator, num_control_points: int = 5
) -> utils.Image:
    """
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber.

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    rng : npr.Generator
        Pseudo-random number generator.
    num_control_points : int, optional
        Number of inflection points to use on the illumination multiplication
        curve.
        The default is 5.

    Returns
    -------
    newimage : 2D numpy array
        Modified image.

    """
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = rng.uniform(low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.arange(image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(
        image,
        np.reshape(
            np.tile(np.reshape(curve, curve.shape + (1,)), (1, image.shape[1])),
            image.shape,
        ),
    )
    # Rescale values to original range:
    newimage = np.interp(
        newimage, (newimage.min(), newimage.max()), (image.min(), image.max())
    )

    return newimage


#%% SEGMENTATION FUNCTIONS:


class SegmentationDataset(tf.keras.utils.Sequence):
    """Dataset used to train the segmentation model."""

    def __init__(
        self,
        dataset: List[Tuple[utils.Image, utils.SegmentationMask, utils.Image]],
        target_size: Tuple[int, int],
        mode: str,
        kw_data_aug: Dict[str, Any],
        crop: bool,
        rng: npr.Generator,
        stack: bool = False,
    ) -> None:
        """
        Creates a new SegmentationDataset.

        Parameters
        ----------
        dataset : List[Tuple[utils.Image, utils.SegmentationMask, utils.Image]]
            Iterable on (image, segmentation mask, weights) tuples.
        target_size : (int, int)
            Target size of the images (input size of the neural network).
        mode : str, "training" or "evaluation"
            In "training" mode, data augmentation is applied.  In "evaluation"
            mode it is not.
        kw_data_aug : Dict[str, Any]
            Parameters for the data augmentation function (see `data_augmentation`).
        crop : bool
            If `True`, the images are cropped (randomly in training mode and
            with overlapping tiles in evaluation mode).  If `False`, the images
            are simply resized to the target size.
        rng : npr.Generator
            Random number generator.
        """
        if mode not in ["training", "evaluation"]:
            raise ValueError('`mode` can only be `"training"` or `"evaluation"`')
        if kw_data_aug and mode == "evaluation":
            raise ValueError("No data augmentation is supported in evaluation mode.")
        images, masks, weights = dataset
        if crop:
            if mode == "training":
                self.images = images
                self.masks = masks
                self.weights = weights
            else:
                images = [utils.create_windows(img, target_size)[0] for img in images]
                masks = [utils.create_windows(seg, target_size)[0] for seg in masks]
                weights = [utils.create_windows(wei, target_size)[0] for wei in weights]
                self.images = np.concatenate(images)
                self.masks = np.concatenate(masks)
                self.weights = np.concatenate(weights)
        else:
            self.images = [
                cv2.resize(img, target_size[::-1], cv2.INTER_LINEAR) for img in images
            ]
            self.masks = [
                cv2.resize(seg, target_size[::-1], cv2.INTER_NEAREST) for seg in masks
            ]
            self.weights = [
                cv2.resize(wei, target_size[::-1], cv2.INTER_LINEAR) for wei in weights
            ]
        self.order = np.arange(len(self.images))
        self.target_size = target_size
        self.mode = mode
        self.kw_data_aug = kw_data_aug
        self.crop = crop
        self.rng = rng
        self.stack = stack
        print(f"{mode} dataset with {len(self)} samples.")

    def __len__(self) -> int:
        """
        Length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Access a given training sample.
        """
        idx = self.order[idx]
        image, mask, weights = self.images[idx], self.masks[idx], self.weights[idx]
        if self.mode == "evaluation":
            if self.stack:
                return (
                    image[np.newaxis, :, :, np.newaxis],
                    np.dstack([mask, weights])[np.newaxis, :, :, :],
                )
            return (
                image[np.newaxis, :, :, np.newaxis],
                mask[np.newaxis, :, :, np.newaxis],
                weights[np.newaxis, :, :, np.newaxis],
            )
        if self.crop:
            image, mask, weights = random_crop(
                image, mask, weights, self.target_size, self.rng
            )
        image, mask, weights = data_augmentation(
            [image, mask, weights],
            aug_par=self.kw_data_aug,
            order=[1, 0, 1],
            rng=self.rng,
        )
        if self.stack:
            return (
                image[np.newaxis, :, :, np.newaxis],
                np.dstack([mask, weights])[np.newaxis, :, :, :],
            )
        return (
            image[np.newaxis, :, :, np.newaxis],
            mask[np.newaxis, :, :, np.newaxis],
            weights[np.newaxis, :, :, np.newaxis],
        )

    def on_epoch_end(self) -> None:
        """
        Method executed after each epoch.
        """
        if self.mode == "training":
            self.rng.shuffle(self.order)


def load_training_dataset_seg(
    dataset_path: Path,
    target_size: Tuple[int, int],
    crop: bool,
    kw_data_aug: Dict[str, Any],
    validation_split: float = 0,
    test_split: float = 0,
    seed: int = 1,
    stack: bool = False,
) -> Union[
    SegmentationDataset,
    Tuple[SegmentationDataset, SegmentationDataset],
    Tuple[SegmentationDataset, SegmentationDataset, SegmentationDataset],
]:
    """
    Creates new segmentation datasets.

    Parameters
    ----------
    dataset_path : Path
        Path of the folder that contains `img`, `seg` and optionally `wei`.
    target_size : (int, int)
        Target size of the images (input size of the neural network).
    crop : bool
        If `True`, the images are cropped to the target size, if `False`, they
        are resized.  Typically we resize mother machine images and crop 2D pad
        images.
    kw_data_aug : Dict[str, Any]
        Parameters for the data augmentation function (see `data_augmentation`).
    validation_split : float, optional
        Proportion (between 0 and 1) of the input images used for the validation set.
        The default is 0.
    test_split : float, optional
        Proportion (between 0 and 1) of the input images used for the test set.
        The default is 0.
    seed : int, optional
        Seed for the random number generator.
        The default is 1.
    stack : bool, optional
        Whether to stack the labels and weights.  This is a technicality that
        should be removed soon.  For now, specify `True` for cell segmentation
        and `False` for RoI segmentation (follow the example scripts).

    Returns
    -------
    train_ds : SegmentationDataset,
    validation_ds : SegmentationDataset, (optional)
    test_ds : SegmentationDataset, (optional)
    """
    rng = npr.default_rng(seed)

    datasets = _load_images_seg(dataset_path, validation_split, test_split, rng)

    train_imgs = datasets[0] if validation_split or test_split else datasets

    train_ds = SegmentationDataset(
        train_imgs,
        target_size,
        mode="training",
        kw_data_aug=kw_data_aug,
        crop=crop,
        rng=rng,
        stack=stack,
    )

    if not validation_split and not test_split:
        return train_ds

    other_dss = [
        SegmentationDataset(
            other_imgs,
            target_size,
            mode="evaluation",
            kw_data_aug=None,
            crop=crop,
            rng=None,
            stack=stack,
        )
        for other_imgs in datasets[1:]
    ]
    return [train_ds, *other_dss]


def _load_images_seg(
    dataset_path: Path,
    validation_split: float,
    test_split: float,
    rng: npr.Generator,
) -> Iterable[Tuple[utils.Image, utils.SegmentationMask, Union[int, utils.Image]]]:
    """
    Loads the training dataset for ROI segmentation, and splits them randomly
    into non-overlapping splits (training, validation, etc.).

    Parameters
    ----------
    dataset_path: Path
        Path of the training dataset, containing `img` (input images) and `seg`
        (segmentation weights).

    Returns
    -------
    datasets: Iterable[Tuple[utils.Image, utils.SegmentationMask, utils.Image]]
        The same number of datasets as requested splits.  Each dataset is a
        tuple of (original image, segmentation mask, sample weights).
    """
    with_weights = (dataset_path / "wei").is_dir()

    counter = Counter()
    images = []
    masks = []
    if with_weights:
        weights = []
    for img_path in sorted((dataset_path / "img").iterdir()):
        if img_path.name == "Sample000245.png":
            continue
        image = utils.read_image(img_path)
        counter[tuple(image.flatten())] += 1
        if counter[tuple(image.flatten())] > 1:
            continue
        images.append(image)
        mask = utils.read_image(dataset_path / "seg" / img_path.name)
        if mask.shape != image.shape:
            raise RuntimeError(
                "Check the dataset. Image and mask don't have the same "
                f"shape for {img_path.name}."
            )
        masks.append(mask.astype(np.uint8))
        if with_weights:
            weight = utils.read_image(dataset_path / "wei" / img_path.name)
            if weight.shape != image.shape:
                raise RuntimeError(
                    "Check the dataset. Image and weights don't have the same "
                    f"shape for {img_path.name}."
                )
            weights.append(weight)

    if any(count > 1 for count in counter.values()):
        warnings.warn("You have duplicate training images. Is this intentional?")

    # Create an array of indices, each of them with a frequency given by its
    # respective split, then shuffle it.
    splits = np.array([0, 1 - validation_split - test_split, 1 - test_split, 1])
    nsplits = np.diff(np.round(splits * len(images)))
    indices = np.concatenate([[i] * int(n) for i, n in enumerate(nsplits)])
    rng.shuffle(indices)

    if with_weights:
        weights = [
            counter[tuple(image.flatten())] * weight
            for image, weight in zip(images, weights)
        ]
    else:
        weights = [
            counter[tuple(image.flatten())] * np.ones_like(image) for image in images
        ]

    flags = [True, validation_split > 0, test_split > 0]
    datasets = [
        (
            [image for index, image in zip(indices, images) if index == i],
            [mask for index, mask in zip(indices, masks) if index == i],
            [weight for index, weight in zip(indices, weights) if index == i],
        )
        for i, flag in enumerate(flags)
        if flag
    ]
    return datasets[0] if len(datasets) == 1 else datasets


def random_crop(
    img: utils.Image,
    seg: utils.SegmentationMask,
    wei: utils.Image,
    target_size: Tuple[int, int],
    rng: npr.Generator,
) -> Tuple[utils.Image, utils.SegmentationMask, utils.Image]:
    """
    Returns a tuple of (image, segmentation mask, weight map) randomly cropped
    and padded with zeros if necessary.

    Parameters
    ----------
    img: utils.Image
        Input image.
    seg: utils.SegmentationMask
        Segmentation mask (binary image with 0 for background and 1 for cells).
    wei: utils.Image
        Pixel-wise weights.
    target_size: Tuple[int, int]
        Target size for the resizing.
    rng: npr.Generator
        Random number generator.

    Returns
    -------
    img, seg, wei: Tuple[utils.Image, utils.SegmentationMask, utils.Image]
        Resized training sample.
    """
    if img.shape == target_size:
        return (img, seg, wei)

    start_y = rng.integers(max(0, img.shape[0] - target_size[0]) + 1)
    start_x = rng.integers(max(0, img.shape[1] - target_size[1]) + 1)
    end_y = min(start_y + target_size[0], img.shape[0])
    end_x = min(start_x + target_size[1], img.shape[1])
    pad_y = target_size[0] - (end_y - start_y)
    pad_x = target_size[1] - (end_x - start_x)
    padding = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2))

    img = np.pad(img[start_y:end_y, start_x:end_x], padding)
    seg = np.pad(seg[start_y:end_y, start_x:end_x], padding)
    wei = np.pad(wei[start_y:end_y, start_x:end_x], padding)

    assert img.shape == target_size, f"{img.shape}"
    assert seg.shape == target_size, f"{seg.shape}"
    assert wei.shape == target_size, f"{wei.shape}"
    return (img, seg, wei)


def saveResult_seg(
    save_path: Path,
    npyfile: np.ndarray,
    files_list: List[Path] = [],
    multipage: bool = False,
) -> None:
    """
    Saves an array of segmentation output images to disk.

    Parameters
    ----------
    save_path : string
        Path to save folder.
    npyfile : 3D or 4D numpy array
        Array of segmentation outputs to save to individual files. If 4D, only
        the images from the first index of axis=3 will be saved.
    files_list : list of strings, optional
        Filenames to save the segmentation masks as. png, tif or jpg extensions
        work.
        The default is [].
    multipage : bool, optional
        Flag to save all output masks as a single, multi-page TIFF file. Note
        that if the file already exists, the masks will be appended to it.
        The default is False.

    Returns
    -------
    None.

    """
    for i, item in enumerate(npyfile):
        if item.ndim == 3:
            img = item[:, :, 0]
        else:
            img = item
        if multipage:
            filename = save_path / files_list[0]
            io.imsave(
                filename, img * 255, plugin="tifffile", append=True
            )
        else:
            if files_list:
                filename = save_path / files_list[i].name
            else:
                filename = save_path / f"{i}_predict.png"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename, img * 255)


def predictGenerator_seg(
    files_path: Path,
    files_list: Optional[Union[List[str], Tuple[str, ...]]] = None,
    target_size: Tuple[int, int] = (256, 32),
    crop_windows: bool = False,
) -> Iterator[np.ndarray]:
    """
    Get a generator for predicting segmentation on new image files
    once the segmentation U-Net has been trained.

    Parameters
    ----------
    files_path : Path
        Path to image files folder.
    files_list : list/tuple of strings, optional
        List of file names to read in the folder. If empty, all
        files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Size for the images to be resized.
        The default is (256,32).
    crop_windows : bool
        TODO
        The default is False.

    Returns
    -------
    mygen : generator
        Generator that will yield single image files as 4D numpy arrays of
        size (1, target_size[0], target_size[1], 1).

    """
    files_list = files_list or sorted(list(files_path.iterdir()))

    def generator(
        files_path: Path,
        files_list: Iterable[Path],
        target_size: Tuple[int, int],
    ) -> Iterator[np.ndarray]:
        for fname in files_list:
            img = readreshape(
                files_path / fname,
                target_size=target_size,
                order=1,
                method="pad" if crop_windows else "resize",
            )
            # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            img = np.reshape(img, (1,) + img.shape)

            yield img

    mygen = generator(files_path, files_list, target_size)
    return mygen


def seg_weights(
    mask: utils.SegmentationMask,
    classweights: Tuple[float, float] = (1, 1),
    w0: float = 12,
    sigma: float = 2,
) -> npt.NDArray[np.float32]:
    """
    This function computes the weight map as described in the original U-Net
    paper to force the model to learn borders
    (Slow, best to run this offline before training).

    Parameters
    ----------
    mask : 2D array
        Training output segmentation mask.
    classweights : tuple of 2 int/floats
        Weights to apply to background, foreground.
        The default is (1,1)
    w0 : int or float, optional
        Base weight to apply to smallest distance (1 pixel).
        The default is 12.
    sigma : int or float, optional
        Exponential decay rate to apply to distance weights.
        The default is 2.

    Returns
    -------
    weightmap : 2D array
        Weights map image.

    """
    # Get cell labels mask
    lblnb, lblimg = cv2.connectedComponents(utils.to_integer_values(mask, np.uint8))

    # Compute cell-to-cell distances
    distances = np.inf * np.ones((*mask.shape, max(lblnb, 2)))
    for i in range(lblnb - 1):
        imask = (lblimg != i + 1).astype(np.uint8)
        distances[:, :, i] = cv2.distanceTransform(
            imask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

    # Put 2 smallest distances at the beginning
    distances.partition(1, axis=-1)

    # Compute weights map
    weights = w0 * np.exp(-distances[:, :, :2].sum(-1) ** 2 / (2 * sigma**2))
    weights += classweights[0]
    weights[mask == 1] = classweights[1]

    return weights


def make_weights(
    training_dataset_path: Union[str, Path],
    weights_function: Callable[[utils.SegmentationMask], npt.NDArray[np.float32]],
) -> None:
    """
    Create the directory `training_dataset_path/"wei"` and populate it with the
    image weights for training.  The directory `training_dataset_path` should
    already contain "img" with the input images and "seg" with the segmentation
    masks (images with 0 for background pixels and 255 for cell pixels).

    Parameters
    ----------
    training_dataset_path : str or Path
        Directory containing two directories: "img" for the images and "seg"
        for the segmentation masks.  A third directory "wei" will be created
        inside for the weights.
    weights_function : Callable[utils.SegmentationMask, npt.NDArray[np.float32]],
        Function that takes a segmentation mask (0 for background and 1 for
        cells) and returns a weight array (non-negative floats).  The weights
        will be normalized by their maximum before being written down as images.
        Possible values are `delta.data.seg_weights` for mothermachine images,
        `delta.data.seg_weights_2D` for 2D pads, or custom functions or lambdas.
    """
    training_dataset_path = Path(training_dataset_path)
    (training_dataset_path / "wei").mkdir()
    for seg_path in sorted((training_dataset_path / "seg").iterdir()):
        print(f"Making weights for {seg_path.name}...")
        mask = utils.read_image(seg_path)
        weights = weights_function(mask)
        norm_weights = weights / weights.max() if weights.max() > 0 else weights
        int_weights = utils.to_integer_values(norm_weights, np.uint8)
        cv2.imwrite(str(training_dataset_path / "wei" / seg_path.name), int_weights)


def kernel(n: int) -> utils.SegmentationMask:
    """
    Get kernel for morphological operations.

    Parameters
    ----------
    n : Int
        Determine size of kernel.

    Returns
    -------
    kernel : Array of unit8
        Returns a kernel with size of n.

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    return kernel


def estimate_seg2D_classweights(
    mask_path: Path,
    sample_size: Optional[int] = None,
    rng: Optional[npr.Generator] = None,
) -> Tuple[float, float]:
    """
    Estimate the weights to assign each class in the weight maps.

    Parameters
    ----------
    mask_path : str
        Path to folder containing segmentations.
    sample_size : int, optional
        Determines the size of the training set that will be used to calculate class weights.
        The default is None.
    rng : npr.Generator, optional
        Pseudo-random number generator used to subsample the dataset if needed.

    Returns
    -------
    class1 : float
        weight of class 1.
    class2 : float
        weight of class 2.

    """
    mask_names = utils.listfiles(mask_path, {".png", ".tif"})

    # Subsample the training set to reduce computation time.
    if sample_size:
        if rng is None:
            rng = npr.default_rng()
        mask_names = rng.choice(mask_names, sample_size)

    c1 = 0
    c2 = 0

    for mask_name in mask_names:

        mask = utils.read_image(mask_name)
        # Extract all pixels that include the cells and its border (no background)
        border = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(20))
        # Set all pixels that include the cells to zero to leave behind the border only
        border[mask > 0] = 0

        # Bkgd (background) is just the image minus the cells + border
        bkgd = np.ones(mask.shape)
        bkgd = bkgd - (mask + border)

        # Erode the segmentation to avoid putting high emphasiss on edges of cells
        mask_erode = cv2.erode(mask, kernel(2))
        mask_dil = cv2.dilate(mask, kernel(3))
        border_erode = (mask_dil < 1) * (border > 0) * 1

        skel = morph.skeletonize(mask_erode > 0) * 1
        skel_border = morph.skeletonize(border_erode > 0) * 1

        c1 = c1 + np.sum(np.sum(skel.astype(np.float64)))
        c2 = c2 + np.sum(np.sum(skel_border.astype(np.float64)))

    if c1 > c2:
        class1 = c2 / c1
        class2 = 1.0
    else:
        class1 = 1.0
        class2 = 1.0 * c1 / c2

    return class1, class2


def seg_weights_2D(
    mask: utils.SegmentationMask, classweights: Tuple[int, int] = (1, 1)
) -> npt.NDArray[np.float32]:
    """
    Compute custom weightmaps designed for bacterial images
    where borders are difficult to distinguish.

    Parameters
    ----------
    mask : 2D array
        Training output segmentation mask.
    classweights : tuple of 2 int/floats, optional
        Weights to apply to cells and border
        The default is (1,1)


    Returns
    -------
    weightmap : 2D array
        Weights map image.

    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if np.max(mask) == 255:
        mask[mask > 0] = 1

    # Extract all pixels that include the cells and it's border
    border = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(20))
    # Set all pixels that include the cells to zero to leave behind the border only
    border[mask > 0] = 0

    # Erode the segmentation to avoid putting high emphasiss on edges of cells
    mask_erode = cv2.erode(mask, kernel(2))

    # Get the skeleton of the segmentation and border
    mask_skel = morph.skeletonize(mask_erode > 0)
    border_skel = morph.skeletonize(border > 0)

    # Find the distances from the skeleton of the segmention and border
    border_dist = cv2.distanceTransform(
        255 * (border_skel < 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    mask_dist = cv2.distanceTransform(
        255 * (mask_skel < 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    # Use the distance from the skeletons to create a gradient towards the skeleton
    border_gra = border * (classweights[1]) / (border_dist + 1) ** 2
    mask_gra = mask / (mask_dist + 1) ** 2

    # Set up weights array
    weightmap = np.zeros((mask.shape), dtype=np.float32)

    # Add the gradients for the segmentation and border into the weights array
    weightmap[mask_erode > 0] = mask_gra[mask_erode > 0]
    weightmap[border > 0] = border_gra[border > 0]

    # Set the skeletons of the segmentation and borders to the maximum values
    weightmap[mask_skel > 0] = classweights[0]
    weightmap[border_skel > 0] = classweights[1]

    # Keep the background zero and set the erode values in the seg/border to a minimum of 1
    bkgd = np.ones(mask.shape) - mask - border
    weightmap[((weightmap == 0) * (bkgd < 1))] = 1 / 255

    return weightmap


def estimateClassweights(gene: Iterator, num_samples: int = 30) -> Tuple[float, ...]:
    """
    Estimate the class weights to use with the weighted
    categorical cross-entropy based on the output of the trainGenerator_track
    output.

    Parameters
    ----------
    gene : generator
        Tracking U-Net training generator. (output of trainGenerator_seg/track)
    num_samples : int, optional
        Number of batches to use for estimation. The default is 30.

    Returns
    -------
    class_weights : tuple of floats
        Relative weights of each class. Note that, if 0 elements of a certain
        class are present in the samples, the weight for this class will be set
        to 0.

    """
    sample = next(gene)
    if sample[1].shape[-1] == 1:
        class_counts = [0, 0]
    else:
        class_counts = [0] * sample[1].shape[-1]  # List of 0s

    # Run through samples and classes/categories:
    for _ in range(num_samples):
        if sample[1].shape[-1] == 1:
            class_counts[1] += np.mean(sample[1] > 0)
            class_counts[0] += np.mean(sample[1] == 0)
        else:
            for i in range(sample[1].shape[-1]):
                class_counts[i] += np.mean(sample[1][..., i])
        sample = next(gene)

    # Warning! If 0 elements of a certain class are present in the samples, the
    # weight for this class will be set to 0. This is for the tracking case
    # (Where there are only empty daughter images in the training set)
    # Try changing the num_samples value if this is a problem

    # Normalize by nb of samples and invert to get weights, unless x == 0 to
    # avoid Infinite weights or errors
    class_weights = tuple(num_samples / x if x != 0 else 0 for x in class_counts)

    return class_weights


#%% TRACKING FUNCTIONS


def trainGenerator_track(
    batch_size: int,
    img_path: Path,
    seg_path: Path,
    previmg_path: Path,
    segall_path: Path,
    track_path: Path,
    weights_path: Optional[Path] = None,
    augment_params: Dict[str, Any] = {},
    crop_windows: bool = False,
    target_size: Tuple[int, int] = (256, 32),
    shift_cropbox: int = 0,
    seed: int = 1,
) -> Iterator[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """
    Generator for training the tracking U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images (current timepoint).
    seg_path : string
        Path to folder containing training 'seed' images, ie mask of 1 cell
        in the previous image to track in the current image.
    previmg_path : string
        Path to folder containing training input images (previous timepoint).
    segall_path : string
        Path to folder containing training 'segall' images, ie mask of all
        cells in the current image.
    track_path : string
        Path to folder containing tracking groundtruth, ie mask of
        the seed cell and its potential daughter tracked in the current frame.
    weights_path : string or None, optional
        Path to folder containing pixel-wise weights to apply to the tracking
        groundtruth. If None, the same weight is applied to all pixels. If the
        string is 'online', weights will be generated on the fly (not
        recommended, much slower)
        The default is None.
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    target_size : tuple of 2 ints, optional
        Input and output image size.
        The default is (256,32).
    crop_windows : bool, optional
        Whether to crop out a window of size `target_size` around the seed/seg
        cell to track for all input images instead of resizing.
    shift_cropbox : int, optional
        If `crop_windows` is True, a shift between [-shift_cropbox, +shift_cropbox]
        will be uniformly sampled for both the X and the Y axis. This shift in
        pixels will be applied only to the the cropbox for the current timepoint
        input frames (img,segall,mot_dau,wei), to simulate image drift over time.
    seed : int, optional
        Seed for numpy's random generator.
        The default is 1.

    Yields
    ------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the U-Net training routine. Dimensions of
        the tensor are (batch_size, target_size[0], target_size[1], 4)
    outputs_arr : 4D numpy array of floats
        Output masks for the U-Net training routine. Dimensions of the tensor
        are (batch_size, target_size[0], target_size[1], 3). The third index
        of axis=3 contains 'background' masks, ie the part of the tracking
        output groundtruth that is not part of the mother or daughter masks
    """
    resize_method = "pad" if crop_windows else "resize"

    # Initialize variables and arrays:
    image_names = utils.listfiles(img_path, {".png", ".tif"})
    X = np.empty((batch_size,) + target_size + (4,), dtype=np.float32)
    Y = np.empty((batch_size,) + target_size + (2,), dtype=np.float32)

    # Create a PNRG
    rng = npr.default_rng(seed)

    while True:

        X[:, :, :, :] = 0
        Y[:, :, :, :] = 0

        for b in range(batch_size):
            # Pick random image file name
            filename = rng.choice(image_names)

            # Read images:
            img: utils.Image = readreshape(
                filename, target_size=target_size, order=1, method=resize_method
            )
            seg: utils.SegmentationMask = readreshape(
                seg_path / filename.name,
                target_size=target_size,
                binarize=True,
                order=0,
                method=resize_method,
            )
            previmg: utils.Image = readreshape(
                previmg_path / filename.name,
                target_size=target_size,
                order=1,
                method=resize_method,
            )
            segall: utils.SegmentationMask = readreshape(
                segall_path / filename.name,
                target_size=target_size,
                binarize=True,
                order=0,
                method=resize_method,
            )

            track: utils.SegmentationMask = readreshape(
                track_path / filename.name,
                target_size=target_size,
                binarize=True,
                order=0,
                method=resize_method,
            )
            if weights_path is not None:
                if weights_path == "online":
                    weights = tracking_weights(track, segall)
                else:
                    weights = readreshape(
                        weights_path / filename.name,
                        target_size=target_size,
                        binarize=False,
                        order=0,
                        method=resize_method,
                    )
            else:
                weights = np.ones_like(track)

            # Data augmentation:
            [img, seg, previmg, segall, track, weights] = data_augmentation(
                [img, seg, previmg, segall, track, weights],
                augment_params,
                order=[1, 0, 1, 0, 0, 0],
                rng=rng,
            )

            if crop_windows:
                cb, fb = utils.gettrackingboxes(seg)
                shape = cast(Tuple[int, int], seg.shape)  # for mypy
                shiftY, shiftX = utils.getshiftvalues(shift_cropbox, shape, cb, rng)
            else:
                cb = utils.CroppingBox.full(seg)
                fb = utils.CroppingBox.full(seg)
                shiftY, shiftX = 0, 0

            # Add into arrays:

            fb.patch(X[b, :, :, 1], cb.crop(seg))
            fb.patch(X[b, :, :, 2], cb.crop(previmg))

            # Shift crop box for current frame inputs
            cb.xtl += shiftX
            cb.xbr += shiftX
            cb.ytl += shiftY
            cb.ybr += shiftY

            fb.patch(X[b, :, :, 0], cb.crop(img))
            fb.patch(X[b, :, :, 3], cb.crop(segall))

            fb.patch(Y[b, :, :, 0], cb.crop(track))
            fb.patch(Y[b, :, :, 1], cb.crop(weights))

        # Yield batch:
        yield X, Y


def getfnfromprototype(
    prototype: str,
    fileorder: str,
    position: Optional[int] = None,
    chamber: Optional[int] = None,
    frame: Optional[int] = None,
    cellnb: Optional[int] = None,
) -> Path:
    """
    Generate full filename for specific frame based on file path,
    prototype, fileorder, and filenamesindexing.

    Parameters
    ----------
    prototype : str
        Filename prototype, written in c-style format. See delta.utils.xpreader
        for more details
    fileorder : str
        Filenames ordering. See delta.utils.xpreader for more details
    position : int, optional
        Position/series index (0-based indexing).
    chamber : int, optional
        Imaging chamber index (0-based indexing).
    frame : int, optional
        Frame/timepoint index (0-based indexing).
    cellnb: int, optional
        Cell index (0-based indexing).

    Returns
    -------
    string
        Filename.

    """
    filenumbers = []

    for i in fileorder:
        if i == "p":
            assert position is not None
            filenumbers.append(position)
        if i == "c":
            assert chamber is not None
            filenumbers.append(chamber)
        if i == "t":
            assert frame is not None
            filenumbers.append(frame)
        if i == "n":
            assert cellnb is not None
            filenumbers.append(cellnb)

    if cellnb is not None:
        prototype = Path(prototype).stem + "Cell%06d" + Path(prototype).suffix

    return Path(prototype % tuple(filenumbers))


def predictCompilefromseg_track(
    img_path: Path,
    seg_path: Path,
    files_list: List[Path],
    target_size: Tuple[int, int] = (256, 32),
    crop_windows: bool = False,
) -> Tuple[
    npt.NDArray[np.float32],
    List[Path],
    List[Tuple[utils.CroppingBox, utils.CroppingBox]],
]:
    """
    Compile an inputs array for tracking prediction with the tracking U-Net,
    directly from U-Net segmentation masks saved to disk.

    Parameters
    ----------
    img_path : Path
        Path to original single-chamber images folder. The filenames are
        expected in the printf format Position%02d_Chamber%02d_Frame%03d.png
    seg_path : Path
        Path to segmentation output masks folder. The filenames must be the
        same as in the img_path folder.
    files_list : [Path], optional
        List of filenames to compile in the img_path and seg_path folders.
    target_size : tuple of 2 ints, optional
        Input and output image size.
        The default is (256,32).
    crop_windows : bool, optional
        TODO
        The default is False.

    Returns
    -------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the tracking U-Net training routine.
        Dimensions of the tensor are (cells_to_track, target_size[0],
        target_size[1], 4), with cells_to_track the number of segmented cells
        in all segmentation masks of the files_list.
    seg_name_list : [Path]
        Filenames to save the tracking outputs as. The printf format is
        Position%02d_Chamber%02d_Frame%03d_Cell%02d.png, with the '_Cell%02d'
        string appended to signal which cell is being seeded/tracked (from top
        to bottom)
    boxes : list of (dict, dict)
        Cropping box and fill box to re-place output prediction masks in the
        original image coordinates.


    """
    seg_name_list = []

    ind = 0

    resize_method = "pad" if crop_windows else "resize"

    # Get digits sequences in first filename
    numstrs = re.findall(r"\d+", files_list[0].name)
    # Get digits sequences in first filename
    charstrs = re.findall(r"\D+", files_list[0].name)

    # Create the string prototype to be used to generate filenames on the fly:
    if len(numstrs) == 3 and len(charstrs) == 4:
        # Order is position, chamber, frame/timepoint
        prototype = (
            f"{charstrs[0]}%0{len(numstrs[0])}d"
            f"{charstrs[1]}%0{len(numstrs[1])}d"
            f"{charstrs[2]}%0{len(numstrs[2])}d"
            f"{charstrs[3]}"
        )
        fileorder = "pct"
    elif len(numstrs) == 2 and len(charstrs) == 3:
        # Order is position, frame/timepoint
        prototype = (
            f"{charstrs[0]}%0{len(numstrs[0])}d"
            f"{charstrs[1]}%0{len(numstrs[1])}d"
            f"{charstrs[2]}"
        )
        fileorder = "pt"
    else:
        raise ValueError(
            "Filename formatting error. See documentation for image sequence formatting"
        )
    fileordercell = fileorder + "n"

    boxes = []
    for item in files_list:
        filename = item.name
        # Get position, chamber & frame numbers:
        cha: Optional[int]
        if fileorder == "pct":
            (pos, cha, fra) = list(map(int, re.findall(r"\d+", filename)))
        elif fileorder == "pt":
            (pos, fra) = list(map(int, re.findall(r"\d+", filename)))
            cha = None

        if fra > 1:

            prevframename = getfnfromprototype(
                prototype, fileorder, position=pos, chamber=cha, frame=fra - 1
            )

            img = readreshape(
                img_path / filename,
                target_size=target_size,
                order=1,
                method=resize_method,
            )
            segall = readreshape(
                seg_path / filename,
                target_size=target_size,
                order=0,
                binarize=True,
                method=resize_method,
            )
            previmg = readreshape(
                img_path / prevframename,
                target_size=target_size,
                order=1,
                method=resize_method,
            )
            prevsegall = readreshape(
                seg_path / prevframename,
                target_size=target_size,
                order=0,
                binarize=True,
                method=resize_method,
            )

            lblimg, lblnb = label(prevsegall, connectivity=1, return_num=True)

            x = np.zeros(shape=(lblnb,) + target_size + (4,), dtype=np.float32)

            for lbl in range(1, lblnb + 1):
                seg = lblimg == lbl
                seg = seg.astype(np.uint8)  # Output is boolean otherwise

                # Cell-centered crop boxes:
                if crop_windows:
                    shape = cast(Tuple[int, int], img.shape)  # for mypy
                    cb, fb = utils.gettrackingboxes(seg, shape)
                else:
                    cb = utils.CroppingBox.full(img)
                    fb = utils.CroppingBox.full(img)
                boxes += [(cb, fb)]

                # Current image
                fb.patch(x[lbl - 1, :, :, 0], cb.crop(img))

                fb.patch(x[lbl - 1, :, :, 1], cb.crop(seg))

                # Previous image
                fb.patch(x[lbl - 1, :, :, 2], cb.crop(previmg))

                # Segmentation of all current cells
                fb.patch(x[lbl - 1, :, :, 3], cb.crop(segall))

                segfilename = getfnfromprototype(
                    prototype,
                    fileordercell,
                    position=pos,
                    chamber=cha,
                    frame=fra - 1,
                    cellnb=lbl,
                )

                seg_name_list.append(segfilename)

            if ind == 0:
                inputs_arr = x
                ind = 1
            else:
                inputs_arr = np.concatenate((inputs_arr, x), axis=0)
    return inputs_arr, seg_name_list, boxes


def tracking_weights(
    track: np.ndarray, segall: utils.SegmentationMask, halo_distance: int = 50
) -> npt.NDArray[np.float32]:
    """
    Compute weights for tracking training sets.

    Parameters
    ----------
    track : 2D array
        Tracking output mask.
    segall : 2D array
        Segmentation mask of all cells in current image.
    halo_distance : int, optional
        Distance in pixels to emphasize other cells from tracked cell.
        The default is 50.

    Returns
    -------
    weights : 2D array
        Tracking weights map.

    """
    # Cell escaped / disappeared:
    if np.max(track) == 0:
        weights = ((segall > 0).astype(np.float32) * 20) + 1
        return weights

    # Distance from the tracked cell:
    _, dist_from = morph.medial_axis(track == 0, return_distance=True)
    dist_from = halo_distance - dist_from
    dist_from[dist_from < 1] = 1

    # Distance from border within cell:
    _, dist_in = morph.medial_axis(track > 0, return_distance=True)

    # Tracked cell skeleton:
    skel = morph.skeletonize(track, method="lee")

    # Tracked Cell weights are distance from edges + skeleton at 255 in the center
    weights = 1 + 41 * (dist_in.astype(np.float32) - dist_in.min()) / dist_in.ptp()
    weights[skel > 0] = 255

    # Rest of the image is weighed according to distance from tracked cell:
    weights += (
        63
        * (dist_from / halo_distance)
        * (segall > 0).astype(np.float32)
        * (track == 0).astype(np.float32)
    )

    weights *= 100
    weights[dist_from < 1] = 1

    return weights


def saveResult_track(
    save_path: Path,
    npyfile: np.ndarray,
    files_list: Optional[List[Path]] = None,
) -> None:
    """
    Save tracking output masks to disk.

    Parameters
    ----------
    save_path : Path
        Folder to save images to.
    npyfile : 4D numpy array
        Array of tracking outputs to save to individual files.
    files_list : tuple/list of strings, optional
        Filenames to save the masks as. Note that the `mother_` and `daughter_`
        prefixes will be added to those names. If None, numbers will be used.
        The default is None.

    Returns
    -------
    None.

    """
    mothers = npyfile[:, :, :, 0]
    for i, mother in enumerate(mothers):
        if files_list:
            filename = save_path / files_list[i].name
        else:
            filename = save_path / f"tracking_{i:09d}.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filename, mother * 255)
