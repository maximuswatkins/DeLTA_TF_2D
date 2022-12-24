"""
Module for the main processing pipeline.

@author: jeanbaptiste
"""

import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from scipy.io import savemat

from delta import utilities as utils
from delta.config import Config
from delta.data import binarizerange, postprocess
from delta.utilities import CroppingBox


class Pipeline:
    """
    Main Pipeline class to process all positions.
    """

    def __init__(
        self,
        xpreader: utils.xpreader,
        config: Config,
        resfolder: Optional[Union[str, Path]] = None,
        reload: bool = False,
    ) -> None:
        """
        Initialize Pipeline.

        Parameters
        ----------
        xpreader : object
            utilities xpreader object.
        resfolder : str or Path, optional
            Path to folder to save results to.
            The default is None.
        reload : bool, optional
            Flag to reload previous position files from resfolder.
            The default is False.

        Returns
        -------
        None.

        """

        self.reader: utils.xpreader = xpreader
        self.positions: List[Position] = []
        self.config = config

        self.models = {model: config.model(model) for model in config.models}

        for arg_index, arg in enumerate(sys.argv):
            if sys.argv[arg_index] == "--resfolder":
                resfolder = sys.argv[arg_index + 1]

        # Create result files folders
        if resfolder is not None:
            self.resfolder = Path(resfolder)
        else:
            xpfile = self.reader.filename
            if xpfile.is_dir():
                self.resfolder = xpfile / "delta_results"
            else:
                self.resfolder = xpfile.with_name(xpfile.stem + "_delta_results")

        self.resfolder.mkdir(exist_ok=True)

        # Initialize position processors:
        for p in range(self.reader.positions):
            self.positions += [
                Position(
                    p,
                    self.reader,
                    self.models,
                    config=self.config,
                )
            ]

        # If reload flag, reload positions from pickle files:
        if reload:
            for p in range(self.reader.positions):
                self.positions[p] = Position.load(self.resfolder / f"Position{p:06d}.pkl")

    def preprocess(
        self,
        positions: Optional[List[int]] = None,
        references: Optional[np.ndarray] = None,
    ) -> None:
        """
        Pre-process positions (Rotation correction, identify ROIs,
        initialize drift correction).

        Parameters
        ----------
        positions : list of int or None, optional
            List of positions to pre-process. If None, all will be run.
            The default is None.
        references : 3D array or None, optional
            Reference images to use to perform pre-processing. If None,
            the first image of each position will be used. Dimensions
            are (positions, size_y, size_x)
            The default is None.

        Returns
        -------
        None.

        """
        # TODO implement ROIs mode selection here instead of config

        # Process positions to run:
        if positions is None:
            positions_torun = list(range(self.reader.positions))
        else:
            positions_torun = positions

        # Run preprocessing:
        for p in positions_torun:
            self.positions[p].preprocess(
                reference=references[p] if isinstance(references, np.ndarray) else None,
                rotation_correction=self.config.rotation_correction,
            )

    def process(
        self,
        positions: Optional[List[int]] = None,
        frames: Optional[List[int]] = None,
        features: Optional[Tuple[str, ...]] = None,
        clear: bool = True,
    ) -> None:
        """
        Run pipeline.

        Parameters
        ----------
        positions : list of int or None, optional
            List of positions to run. If None, all positions are run.
            The default is None.
        frames : list of int or None, optional
            List of frames to run. If None, all frames are run.
            The default is None.
        features : list of str or None, optional
            List of features to extract. If None, all features are extracted.
            The default is None.
        clear : bool, optional
            Clear variables of each Position object after it has been processed
            and saved to disk, to prevent memory issues.
            The default is True.

        Returns
        -------
        None.

        """
        if frames is None:
            frames = list(range(self.reader.timepoints))

        if positions is None:
            positions = list(range(self.reader.positions))

        if features is None:
            features_list = ["length", "width", "area", "perimeter", "edges"]
            for c in range(1, self.reader.channels):
                features_list += [f"fluo{c}"]
            features = tuple(features_list)

        # Run through positions:
        for p in positions:

            # Preprocess is not done already:
            if not self.positions[p]._preprocessed:
                self.positions[p].preprocess(
                    rotation_correction=self.config.rotation_correction
                )

            # Segment all frames:
            self.positions[p].segment(frames=frames)

            # Track cells:
            self.positions[p].track(frames=frames)

            # Extract features:
            self.positions[p].features(frames=frames, features=features)

            # Save to disk and clear memory:
            self.positions[p].save(
                filename=self.resfolder / f"Position{p:06d}",
                frames=frames,
                save_format=self.config.save_format,
            )

            if clear:
                self.positions[p].clear()

    def run(self) -> None:
        """
        On-the-fly processor (not functional yet).

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not self.on_the_fly:
            raise RuntimeError(
                "This Pipeline was not initialized for on-the-fly processing"
            )

        while True:

            # Check if new files are available:
            newfiles = self.reader.watcher.newfiles
            while len(newfiles) == 0:
                newfiles = self.reader.watcher.newfiles()

            for nf in newfiles:
                pos, chan = nf
                frame = self.reader.watcher.old[pos, chan]

                # Segmentation & tracking for trans images:
                if chan == 0:
                    self.positions[pos].segment(frames=[frame])
                    self.positions[pos].track(frames=[frame])
                    self.positions[pos].features(
                        frames=[frame], features=("length", "width", "area")
                    )

                # Features extraction for fluorescence images:
                else:
                    self.positions[pos].features(
                        frames=[frame], features=[f"fluo{chan}"]
                    )


class Position:
    """
    Position processing object.
    """

    def __init__(
        self,
        position_nb: int,
        reader: utils.xpreader,
        models: Dict[str, tf.keras.Model],
        config: Config,
    ) -> None:
        """
        Initialize Position.

        Parameters
        ----------
        position_nb : int
            Position index.
        reader : object
            utilities xpreader object.
        models : Dict[str, tf.keras.Model]
            U-Net models.
        drift_correction : bool, optional
            Flag to perform drift correction. The default is True.
        crop_windows : bool, optional
            Flag to crop out windows. The default is False.

        Returns
        -------
        None.

        """
        self.position_nb = position_nb
        "Position index number in the experiment"
        self.reader: utils.xpreader = reader
        "Experiment reader object"
        self.models = models
        "List of Tensorflow models"
        self.rois: List[ROI] = []
        "List of ROI objects under position"
        self.drift_values: List = [[], []]
        "XY drift correction values over time"
        self.config = config
        self.verbose: int = 1
        "Console output verbosity"
        self._preprocessed: bool = False
        self._pickle_skip: Tuple[str, ...] = ("reader", "models", "_pickle_skip")

    def __eq__(self, other) -> bool:
        eq_position_nb = self.position_nb == other.position_nb
        eq_rois = self.rois == other.rois
        eq_drift_values = np.array_equal(self.drift_values, other.drift_values)
        eq_verbose = self.verbose == other.verbose
        eq__preprocessed = self._preprocessed == other._preprocessed
        eqs = [
            eq_position_nb,
            eq_rois,
            eq_drift_values,
            eq_verbose,
            eq__preprocessed,
        ]
        return all(eqs)

    def __getstate__(self) -> Dict[str, Any]:
        """
        For pickle.

        Returns
        -------
        state : dict
            Values to store in pickle file.

        """
        return {k: v for k, v in self.__dict__.items() if k not in self._pickle_skip}

    def _msg(self, string: str) -> None:
        """
        Print timestamped messages.

        Parameters
        ----------
        string : str
            Message to print.

        Returns
        -------
        None.

        """
        if self.verbose:
            print(f"{time.ctime()}, Position {self.position_nb} - {string}", flush=True)

    def preprocess(
        self,
        reference: Optional[utils.Image] = None,
        rotation_correction: Union[float, bool] = True,
    ) -> None:
        """
        Pre-process position (Rotation correction, identify ROIs,
        initialize drift correction).

        Parameters
        ----------
        reference : 2D array, optional
            Reference image to use to perform pre-processing. If None,
            the first image of each position will be used.
            The default is None.
        rotation_correction : float or bool, optional
            If bool, flag whether to perform rotation correction.
            If number, value of rotation correction.  The default is True.

        Returns
        -------
        None.

        """
        self._msg("Starting pre-processing")

        # If no reference frame provided, read first frame from reader:
        if reference is None:
            reference = self.reader.getframes(
                position=self.position_nb,
                frames=0,
                channels=0,
                squeeze_dimensions=True,
            )

        # Estimate rotation:
        if isinstance(rotation_correction, bool):
            if rotation_correction:
                self.rotate = utils.deskew(reference)  # Estimate rotation angle
                self._msg(f"Rotation correction: {self.rotate} degrees")
                reference = utils.imrotate(reference, self.rotate)
            else:
                self.rotate = 0
        else:
            self.rotate = rotation_correction

        # Find rois, filter results, get bounding boxes:
        if "rois" in self.models:
            self.detect_rois(reference)
        else:
            self.rois = [
                ROI(
                    roi_nb=0,
                    box=CroppingBox(
                        xtl=0, ytl=0, xbr=reference.shape[1], ybr=reference.shape[0]
                    ),
                    config=self.config,
                )
            ]

        # Get drift correction template and box
        if self.config.drift_correction:
            self.drifttemplate = utils.getDriftTemplate(
                [r.box for r in self.rois],
                reference,
                whole_frame=self.config.whole_frame_drift,
            )
            self.driftcorbox = CroppingBox(
                xtl=0,
                xbr=reference.shape[1],
                ytl=0,
                ybr=reference.shape[0]
                if self.config.whole_frame_drift
                else max(roi.box.ytl for roi in self.rois),
            )

        self._preprocessed = True

    def detect_rois(self, reference: utils.Image) -> None:
        """
        Use U-Net to detect ROIs (chambers etc...).

        Parameters
        ----------
        reference : 2D array
            Reference image to use to perform pre-processing

        Returns
        -------
        None.

        """

        # Rescale pixel values between 0 and 1 for the old model
        reference = (reference - reference.min()) / reference.ptp()

        # Predict
        rois_pred = self.models["rois"].predict(
            cv2.resize(reference, self.config.target_size_rois)[np.newaxis, :, :, np.newaxis],
            verbose=0,
        )

        # Clean up:
        self.roismask = postprocess(
            cv2.resize(np.squeeze(rois_pred), reference.shape[::-1]),
            min_size=self.config.min_roi_area,
        )

        # Get boxes
        roisboxes = utils.getROIBoxes(self.roismask)

        # Instantiate ROIs:
        for b, box in enumerate(roisboxes):
            self.rois += [ROI(roi_nb=b, box=box, config=self.config)]

    def segment(self, frames: List[int]) -> None:
        """
        Segment cells in all ROIs in position.

        Parameters
        ----------
        frames : list of int
            List of frames to run.

        Returns
        -------
        None.

        """
        self._msg(f"Starting segmentation ({len(frames)} frames)")

        # Load trans frames:
        trans_frames = self.reader.getframes(
            position=self.position_nb,
            channels=0,
            frames=frames,
            rotate=self.rotate,
        )

        # If trans_frames is 2 dimensions, an extra dimension is added at axis=0 for time
        # (1 timepoint may cause this issue)
        if trans_frames.ndim == 2:
            trans_frames = trans_frames[np.newaxis, :, :]

        # Correct drift:
        if self.config.drift_correction:
            trans_frames, self.drift_values = utils.driftcorr(
                trans_frames, template=self.drifttemplate, box=self.driftcorbox
            )

        # Run through frames and ROIs and compile segmentation inputs:
        x = []
        references: List[Tuple[int, int, int, Optional[Tuple[List, List]]]] = []
        for f, img in enumerate(trans_frames):
            for r, roi in enumerate(self.rois):
                inputs, windows = roi.get_segmentation_inputs(img)
                x += [inputs]
                references += [(r, len(x[-1]), f, windows)]
        x = np.concatenate(x)

        # Run segmentation model:
        y = np.empty_like(x)
        for i in range(0, len(x), self.config.pipeline_seg_batch):
            self._msg(f"Segmentation - ROI ({i}/{len(x)})")
            j = min((len(x), i + self.config.pipeline_seg_batch))
            y[i:j] = self.models["segmentation"].predict(x[i:j], batch_size=1)

        # Dispatch segmentation outputs to rois:
        i = 0
        for ref in references:
            self.rois[ref[0]].process_segmentation_outputs(
                y[i : i + ref[1]], frame=ref[2], windows=ref[3]
            )
            i += ref[1]

    def track(self, frames: List[int]) -> None:
        """
        Track cells in all ROIs in frames.

        Parameters
        ----------
        frames : list of int
            List of frames to run.

        Returns
        -------
        None.

        """
        self._msg(f"Starting tracking ({len(frames)} frames)")

        for frame in frames:
            self._msg(f"Tracking - frame {frame}/{len(frames)} ")

            # Compile inputs and references:
            x = []
            references = []
            for iroi, roi in enumerate(self.rois):
                inputs, boxes = roi.get_tracking_inputs(frame=frame)
                x.append(inputs)
                references.append((iroi, len(inputs), boxes))

            # Predict:
            x = (
                np.concatenate(x)
                if x
                else np.empty((0, *self.config.target_size_track, 4))
            )
            y = np.empty(shape=(*x.shape[:3], 1))
            for i in range(0, len(x), self.config.pipeline_track_batch):
                j = min(len(x), i + self.config.pipeline_track_batch)
                y[i:j] = self.models["tracking"].predict(
                    x[i:j],
                    batch_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    verbose=0,
                )

            # Dispatch tracking outputs to rois:
            i = 0
            for iroi, nbcells, boxes in references:
                self.rois[iroi].process_tracking_outputs(
                    y[i : i + nbcells], frame=frame, boxes=boxes
                )
                i += nbcells

    def features(
        self,
        frames: List[int],
        features: Tuple[str, ...] = (
            "length",
            "width",
            "area",
            "perimeter",
            "edges",
        ),
    ) -> None:
        """
        Extract features for all ROIs in frames.

        Parameters
        ----------
        frames : list of int
            List of frames to run.
        features : list or tuple of str, optional
            List of features to extract.
            The default is ("length", "width", "area", "perimeter", "edges").

        Returns
        -------
        None.

        """
        self._msg(f"Starting feature extraction ({len(frames)} frames)")

        # Check if fluo channels are requested in features (fluo1, fluo2, etc):
        fluo_channels = [int(x[4:]) for x in features if x[0:4] == "fluo"]

        # Load fluo images if any:
        if len(fluo_channels):
            # Read fluorescence frames:
            fluo_frames = self.reader.getframes(
                position=self.position_nb,
                channels=fluo_channels,
                frames=frames,
                squeeze_dimensions=False,
                rotate=self.rotate,
            )
            # Apply drift correction
            if self.config.drift_correction:
                for f in range(fluo_frames.shape[1]):
                    fluo_frames[:, f, :, :], _ = utils.driftcorr(
                        fluo_frames[:, f, :, :], drift=self.drift_values
                    )
        else:
            fluo_frames = None

        # Run through frames:
        for f in frames:
            self._msg(f"Feature extraction - frame {f}/{len(frames)}")
            # Run through ROIs and extract features:
            for roi in self.rois:
                roi.extract_features(
                    frame=f,
                    fluo_frames=fluo_frames[f] if fluo_frames is not None else None,
                    features=features,
                )

        # Now extracting multi-frame features
        for roi in self.rois:
            for cellid, cell in roi.lineage.cells.items():
                for iframe in cell["frames"]:
                    for feature in ["length", "area"]:
                        try:
                            gr = roi.lineage.growthrate(cellid, iframe, feature)
                            roi.lineage.setvalue(cellid, iframe, f"growthrate_{feature}", gr)
                        except KeyError:
                            warnings.warn(
                                "An issue happened related to feature extraction: "
                                "some cells cannot be found after resizing."
                            )

    def save(
        self,
        filename: Optional[Union[str, Path]] = None,
        frames: Optional[List[int]] = None,
        save_format: Tuple[str, ...] = ("pickle", "movie"),
    ) -> None:
        """
        Save to disk.

        Parameters
        ----------
        filename : str or None, optional
            File name for save file. If None, the file will be saved to
            PositionXXXXXX in the current directory.
            The default is None.
        frames : list of int or None, optional
            List of frames to save in movie. If None, all frames are run.
            The default is None.
        save_format : tuple of str, optional
            Formats to save the data to. Options are "pickle', 'legacy' (ie
            Matlab format), and "movie' for saving an mp4 movie.
            The default is ("pickle', 'movie').

        Returns
        -------
        None.

        """
        if filename is None:
            filename = f"./Position{self.position_nb:06d}"
        filename = Path(filename)

        if "legacy" in save_format:
            self._msg("Saving to legacy format\n" + str(filename.with_suffix(".mat")))
            self.legacysave(filename.with_suffix(".mat"))

        if "pickle" in save_format:
            self._msg("Saving to pickle format\n" + str(filename.with_suffix(".pkl")))
            with open(filename.with_suffix(".pkl"), "wb") as file:
                pickle.dump(self, file)

        if "movie" in save_format:
            self._msg("Saving results movie\n" + str(filename.with_suffix(".mp4")))
            movie = self.results_movie(frames=frames)
            utils.vidwrite(movie, filename.with_suffix(".mp4"), verbose=False)

    @staticmethod
    def load(filename: Union[str, Path]):
        """
        Load position from pickle file.

        Parameters
        ----------
        filename : str or Path
            File name for save file.

        Returns
        -------
        position : pipeline.Position object
            Reloaded Position object. (without reader and models)

        """

        with open(filename, "rb") as file:
            position = pickle.load(file)

        # Retrocompatibility with old format
        for roi in position.rois:
            if isinstance(roi.lineage.cells, list):
                roi.lineage.cells = {cellid: cell for cellid, cell in enumerate(roi.lineage.cells)}

        return position

    def clear(self) -> None:
        """
        Clear Position-specific variables from memory (can be loaded back with
        load()).

        Returns
        -------
        None.

        """
        self._msg("Clearing variables from memory")
        for k in self.__dict__:
            if k not in self._pickle_skip:
                setattr(self, k, None)

    def legacysave(self, res_file: Union[str, Path]) -> None:
        """
        Save pipeline data in the legacy Matlab format.

        Parameters
        ----------
        res_file : str or Path
            Path to save file.

        Returns
        -------
        None.

        """
        # File reader info
        moviedimensions = [
            self.reader.y,
            self.reader.x,
            self.reader.channels,
            self.reader.timepoints,
        ]
        xpfile = self.reader.filename

        # If No ROIs detected for position:
        if len(self.rois) == 0:
            savemat(
                res_file,
                {
                    "res": [],
                    "tiffile": str(xpfile),
                    "moviedimensions": moviedimensions,
                    "proc": {"rotation": self.rotate, "chambers": [], "XYdrift": []},
                },
            )
            return

        # Initialize data structure/dict:
        data = dict(moviedimensions=moviedimensions, tifffile=str(xpfile))

        # Proc dict/structure:
        data["proc"] = dict(
            rotation=self.rotate,
            XYdrift=np.array(self.drift_values, dtype=np.float64),
            chambers=np.array(
                [
                    [
                        r.box.xtl,
                        r.box.ytl,
                        r.box.xbr - r.box.xtl,
                        r.box.ybr - r.box.ytl,
                    ]
                    for r in self.rois
                ],
                dtype=np.float64,
            ),
        )

        # Lineages:
        data["res"] = []
        for r in self.rois:
            res: Dict[str, Any] = {}

            # Resized labels stack: (ie original ROI size)
            res["labelsstack_resized"] = np.array(r.label_stack, dtype=np.uint16)

            # Not resized stack: (ie U-Net seg target size)
            label_stack = []
            for f, cellnbs in enumerate(r.lineage.cellnumbers):
                label_stack += [
                    utils.label_seg(r.seg_stack[f], [c + 1 for c in cellnbs])
                ]
            res["labelsstack"] = np.array(label_stack, dtype=np.uint16)

            # Run through cells, update to 1-based indexing
            lin: List[Dict[str, Any]] = []
            for cell in r.lineage.cells.values():
                lin += [{}]
                # Base lineage:
                lin[-1]["mother"] = cell["mother"] + 1 if cell["mother"] is not None else 0
                lin[-1]["frames"] = np.array(cell["frames"], dtype=np.float32) + 1
                lin[-1]["daughters"] = np.array(cell["daughters"], dtype=np.float32) + 1
                lin[-1]["daughters"][np.isnan(lin[-1]["daughters"])] = 0
                if "edges" in cell:
                    lin[-1]["edges"] = cell["edges"]

                # Morphological features:
                morpho_features = (
                    "area",
                    "width",
                    "length",
                    "perimeter",
                    "old_pole",
                    "new_pole",
                    "growthrate_length",
                    "growthrate_area",
                )
                for feat in morpho_features:
                    if feat in cell:
                        lin[-1][feat] = np.array(cell[feat], dtype=np.float32)

                # Loop through potential fluo channels:
                fluo = 0
                while True:
                    fluo += 1
                    fluostr = f"fluo{fluo}"
                    if fluostr in cell:
                        lin[-1][fluostr] = np.array(cell[fluostr], dtype=np.float32)
                    else:
                        break
            # Store into res dict:
            res["lineage"] = lin

            # Store into data structure:
            data["res"] += [res]

        # Finally, save to disk:
        savemat(res_file, data)

    def results_movie(self, frames: Optional[List[int]] = None) -> Any:
        """
        Generate movie illustrating segmentation and tracking.

        Parameters
        ----------
        frames : list of int or None, optional
            Frames to generate the movie for. If None, all frames are run.
            The default is None.

        Returns
        -------
        movie : list of 3D numpy arrays
            List of compiled movie frames

        """
        # Re-read trans frames:
        trans_frames = self.reader.getframes(
            position=self.position_nb,
            channels=0,
            frames=frames,
            squeeze_dimensions=False,
            rotate=self.rotate,
        )
        trans_frames = trans_frames[:, 0]

        if self.config.drift_correction:
            trans_frames, _ = utils.driftcorr(trans_frames, drift=self.drift_values)
        movie = []

        assert isinstance(frames, list)  # FIXME: what happens if frames is None?
        # Run through frames, compile movie:
        for f, fnb in enumerate(frames):

            frame = trans_frames[f]

            # RGB-ify:
            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)

            # Add frame number text:
            frame = cv2.putText(
                frame,
                text=f"frame {fnb:06d}",
                org=(int(frame.shape[0] * 0.05), int(frame.shape[0] * 0.97)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 1, 1, 1),
                thickness=2,
            )

            for r, roi in enumerate(self.rois):

                # Get chamber-specific variables:
                colors = utils.getrandomcolors(roi.lineage.cells.keys(), seed=r)
                fr = roi.label_stack[fnb]
                assert fr is not None  # FIXME: why is it not None?
                cells, contours = utils.getcellsinframe(fr, return_contours=True)
                assert isinstance(cells, list)  # needed for mypy on Python < 3.8

                if roi.box is None:
                    xtl, ytl = (0, 0)
                else:
                    xtl, ytl = (roi.box.xtl, roi.box.ytl)

                # Run through cells in labelled frame:
                for c, cell in enumerate(cells):

                    # Draw contours:
                    frame = cv2.drawContours(
                        frame,
                        contours,
                        c,
                        color=colors[cell],
                        thickness=1,
                        offset=(xtl, ytl),
                    )

                    # Draw poles:
                    oldpole = roi.lineage.getvalue(cell, fnb, "old_pole")
                    assert isinstance(oldpole, np.ndarray)  # for mypy
                    frame = cv2.drawMarker(
                        frame,
                        (oldpole[1] + xtl, oldpole[0] + ytl),
                        color=colors[cell],
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=3,
                        thickness=1,
                    )

                    daughter = roi.lineage.getvalue(cell, fnb, "daughters")
                    bornago = roi.lineage.cells[cell]["frames"].index(fnb)
                    mother = roi.lineage.cells[cell]["mother"]

                    if daughter is None and (bornago > 0 or mother is None):
                        newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                        frame = cv2.drawMarker(
                            frame,
                            (newpole[1] + xtl, newpole[0] + ytl),
                            color=[1, 1, 1],
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=3,
                            thickness=1,
                        )

                    # Plot division arrow:
                    if daughter is not None:

                        newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                        daupole = roi.lineage.getvalue(daughter, fnb, "new_pole")
                        # Plot arrow:
                        frame = cv2.arrowedLine(
                            frame,
                            (newpole[1] + xtl, newpole[0] + ytl),
                            (daupole[1] + xtl, daupole[0] + ytl),
                            color=(1, 1, 1),
                            thickness=1,
                        )

            # Add to movie array:
            movie += [(frame * 255).astype(np.uint8)]

        return movie


class ROI:
    """
    ROI processor object.
    """

    def __init__(self, roi_nb: int, box: CroppingBox, config: Config) -> None:
        """
        Initialize ROI.

        Parameters
        ----------
        roi_nb : int
            ROI index.
        box : CroppingBox
            CroppingBox for ROI.
        config : Config

        Returns
        -------
        None.

        """
        self.roi_nb = roi_nb
        "The ROI index number"
        self.box = box
        "ROI crop box"
        self.img_stack: List[utils.Image] = []
        "Input images stack"
        self.seg_stack: List[utils.SegmentationMask] = []
        "Segmentation images stack"
        self.lineage = utils.Lineage()
        "Lineage object for ROI"
        self.label_stack: List[utils.Labels] = []
        "Labelled images stack"
        self.config = config

        if config.crop_windows:
            self.scaling = None
        else:
            self.scaling = (
                (box.ybr - box.ytl) / self.config.target_size_seg[0],
                (box.xbr - box.xtl) / self.config.target_size_seg[1],
            )

    def __eq__(self, other) -> bool:
        print(f"ROI {self.roi_nb}")
        eq_roi_nb = self.roi_nb == other.roi_nb
        if not eq_roi_nb:
            print("ROIs differ by roi_nb")
        eq_box = self.box == other.box
        if not eq_box:
            print("ROIs differ by box")
            print(self.box)
            print(other.box)
        eq_img_stack = np.allclose(self.img_stack, other.img_stack, atol=2e-4)
        if not eq_img_stack:
            print("ROIs differ by img_stack")
            try:
                np.testing.assert_allclose(self.img_stack, other.img_stack, atol=2e-4)
            except AssertionError as err:
                print(err)
        eq_seg_stack = np.array_equal(self.seg_stack, other.seg_stack)
        if not eq_seg_stack:
            print("ROIs differ by seg_stack")
            try:
                np.testing.assert_array_equal(self.seg_stack, other.seg_stack)
            except AssertionError as err:
                print(err)
        eq_label_stack = np.array_equal(self.label_stack, other.label_stack)
        if not eq_label_stack:
            print("ROIs differ by label_stack")
        eq_lineage = self.lineage == other.lineage
        if not eq_lineage:
            print("ROIs differ by lineage")
        eqs = [
            eq_lineage,
            eq_roi_nb,
            eq_box,
            eq_img_stack,
            eq_seg_stack,
            eq_label_stack,
        ]
        return all(eqs)

    def get_segmentation_inputs(
        self, img: utils.Image
    ) -> Tuple[np.ndarray, Optional[Tuple[List, List]]]:
        """
        Compile segmentation inputs for ROI.

        Parameters
        ----------
        img : 2D array
            Single frame to crop and send for segmentation.

        Returns
        -------
        x : 4D array
            Segmentation input array. Dimensions are
            (windows, *self.config.target_size_seg, 1).
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        """
        # Crop and scale:
        i = self.box.crop(img)
        # Rescale pixel values between 0 and 1 for the old model
        i = (i - i.min()) / i.ptp()
        # Append i as is to input images stack:
        self.img_stack.append(i)

        if self.config.crop_windows:
            # Crop out windows:
            x, windows_y, windows_x = utils.create_windows(
                i, target_size=self.config.target_size_seg
            )
            windows = (windows_y, windows_x)
            # Shape x to expected format:
            x = x[:, :, :, np.newaxis]
            return x, windows
        # Resize to unet input size (cv2 wants the reverse size)
        x = cv2.resize(i, dsize=self.config.target_size_seg[::-1])
        # Shape x to expected format:
        x = x[np.newaxis, :, :, np.newaxis]
        return x, None

    def process_segmentation_outputs(
        self,
        y: npt.NDArray[np.uint8],
        frame: Optional[int] = None,
        windows: Optional[Tuple[List, List]] = None,
    ) -> None:
        """
        Process outputs after they have been segmented.

        Parameters
        ----------
        y : 4D array
            Segmentation output array. Dimensions are
            (windows, *self.config.target_size_seg, 1).
        frame : int or None, optional
            Frame index. If None, this is considered the latest frame's output.
            The default is None.
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        Returns
        -------
        None.

        """
        # Stitch windows back together (if needed):
        if windows is None:
            y = y[0, :, :, 0]
        else:
            y = utils.stitch_pic(y[..., 0], windows[0], windows[1])

        # Binarize:
        y = binarizerange(y, threshold=0).astype(np.uint8)
        # Crop out segmentation if image was smaller than target_size
        if self.config.crop_windows:
            y = y[: self.img_stack[0].shape[0], : self.img_stack[0].shape[1]]
        # Area filtering:
        y = utils.opencv_areafilt(y, min_area=self.config.min_cell_area)

        # Append to segmentation results stack:
        if frame is None or frame == len(self.seg_stack):
            self.seg_stack.append(y)
        else:
            assert frame < len(self.seg_stack)
            self.seg_stack[frame] = y

    def get_tracking_inputs(
        self, frame: int
    ) -> Tuple[np.ndarray, List[Tuple[CroppingBox, CroppingBox]]]:
        """
        Compile tracking inputs for ROI from `seg_stack`.

        Parameters
        ----------
        frame : int
            The frame to compile for.

        Raises
        ------
        RuntimeError
            Segmentation has not been completed up to frame yet.

        Returns
        -------
        x : 4D array or None
            Tracking input array. Dimensions are
            (previous_cells, *self.config.target_size_track, 4).
        boxes : List of tuples of 2 dicts or None
            Crop and fill boxes to re-place outputs in the ROI

        """

        # Check if segmentation data is ready:
        if len(self.seg_stack) <= frame:
            raise RuntimeError(f"Segmentation incomplete - frame {frame}")

        # Get cell contours from previous frame
        if frame == 0 or len(self.lineage.cellnumbers) < frame:
            prev_cell_contours = []
        else:
            prev_cell_contours = utils.find_contours(self.seg_stack[frame - 1])

        # Sort them along Y
        prev_cell_contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))

        # Allocate empty tracking inputs array:
        x = np.empty((len(prev_cell_contours), *self.config.target_size_track, 4), dtype=np.float32)

        # Run through contours and compile inputs:
        boxes = []
        for c, cell in enumerate(prev_cell_contours):

            draw_offset: Optional[Tuple[float, float]]
            if self.config.crop_windows:
                curr_img = self.img_stack[frame]
                prev_img = self.img_stack[frame - 1]
                # Cell-centered crop boxes:
                shape = cast(Tuple[int, int], curr_img.shape)  # for mypy
                cb, fb = utils.gettrackingboxes(cell, shape, self.config.target_size_track)
                draw_offset = (-cb.xtl + fb.xtl, -cb.ytl + fb.ytl)
            else:
                curr_img = cv2.resize(
                    self.img_stack[frame], dsize=self.config.target_size_seg[::-1]
                )
                prev_img = cv2.resize(
                    self.img_stack[frame - 1], dsize=self.config.target_size_seg[::-1]
                )
                cb = CroppingBox.full(curr_img)
                fb = CroppingBox.full(curr_img)
                draw_offset = None
            boxes += [(cb, fb)]

            # Current image
            fb.patch(x[c, :, :, 0], cb.crop(curr_img))

            # Segmentation mask of one previous cell (seed)
            x[c, :, :, 1] = cv2.drawContours(
                np.zeros(self.config.target_size_track, dtype=np.float32),
                [cell],
                0,
                offset=draw_offset,
                color=1.0,
                thickness=-1,
            )

            # Previous image
            fb.patch(x[c, :, :, 2], cb.crop(prev_img))

            # Segmentation of all current cells
            fb.patch(x[c, :, :, 3], cb.crop(self.seg_stack[frame]))

        # Return tracking inputs and crop and fill boxes:
        return x, boxes

    def process_tracking_outputs(
        self,
        y: np.ndarray,
        frame: int,
        boxes: Optional[List[Tuple[CroppingBox, CroppingBox]]] = None,
    ) -> None:
        """
        Process output from tracking U-Net.

        Get poles, update lineage and create label_stack.

        Parameters
        ----------
        y : 4D array
            Tracking output array. Dimensions are
            (previous_cells, *self.config.target_size_track, 1).
        frame : int
            The frame to process for.
        boxes : List of tuples of 2 dicts or None
            Crop and fill boxes to re-place outputs in the ROI

        Returns
        -------
        None.

        """

        # Get scores and attributions:
        # Label frame but numbered 1, 2, 3, 4, etc. (temporary labels)
        labels = utils.label_seg(self.seg_stack[frame])
        assert isinstance(labels, np.ndarray)  # needed for mypy on Python < 3.8
        scores = utils.getTrackingScores(
            labels, y[:, :, :, 0], boxes=boxes if self.config.crop_windows else None
        )

        attributions = utils.getAttributions(scores)

        # Get poles:
        poles = utils.getpoles(self.seg_stack[frame], labels, scaling=self.scaling)

        # Update lineage:
        self.lineage.update(None, frame)
        # Go through old cells:
        for o in range(attributions.shape[0]):
            attrib = attributions[o, :].nonzero()[0]
            new_cells_poles = []
            for n in attrib:
                new_cells_poles += [poles[n]]
            self.lineage.update(o, frame, attrib=attrib, poles=new_cells_poles)
        # Go through "orphan" cells:
        for n in range(attributions.shape[1]):
            attrib = attributions[:, n].nonzero()[0]
            new_cells_poles = [poles[n]]
            if len(attrib) == 0:
                self.lineage.update(None, frame, attrib=[n], poles=new_cells_poles)

        # Recompile label frame with new labels
        if len(self.lineage.cellnumbers) <= frame:
            cell_nbs = []
        else:
            cell_nbs = [c + 1 for c in self.lineage.cellnumbers[frame]]
        labels = utils.label_seg(self.seg_stack[frame], cell_nbs)

        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)

        assert isinstance(labels, np.ndarray)  # needed for mypy on Python < 3.8
        assert len(self.label_stack) == frame
        self.label_stack.append(labels)

    def extract_features(
        self,
        frame: Optional[int] = None,
        fluo_frames: Optional[np.ndarray] = None,
        features: Tuple[str, ...] = (
            "length",
            "width",
            "area",
            "perimeter",
            "edges",
        ),
    ) -> None:
        """
        Extract single cell features.

        Parameters
        ----------
        frame : int, optional
            The frame to extract for. If None, the earliest frame not yet
            extracted is run.
            The default is None.
        fluo_frames : 3D array, optional
            Fluorescent images to extract fluo from. Dimensions are
            (channels, size_y, size_x).
            The default is None.
        features : list or tuple of str, optional
            Features to extract. Options are ("length", "width", "area", "fluo1",
            "fluo2", "fluo3"...)
            The default is ("length", "width", "area", "perimeter", "edges").

        Returns
        -------
        None.

        """
        # Default frame:
        if frame is None:
            frame = len(self.label_stack)

        if len(self.label_stack) <= frame:
            raise ValueError(f"Asked for frame {frame} but label_stack has only {len(self.label_stack)} frames")

        # Extract features for all cells in the ROI:
        cells, extracted_features = utils.roi_features(
            self.label_stack[frame],
            features=features,
            fluo_frames=fluo_frames,
            roi_box=self.box,
        )

        # Add feature values to lineage for each cell:
        for cell, cell_features in zip(cells, extracted_features):
            for feature_name, feature_val in cell_features.items():
                self.lineage.setvalue(cell, frame, feature_name, feature_val)
