"""
Module for the main processing pipeline.

@author: jeanbaptiste
"""

import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from scipy.io import savemat

from delta import utilities as utils
from delta.config import Config
from delta.data import binarizerange, postprocess
from delta.utilities import CroppingBox
from collections import defaultdict

MODEL = Literal["rois", "seg", "track"]

from time import perf_counter
import tensorflow as tf
import csv

def sync_tf(): # for syncing GPU during inference time measurements
    try:
        tf.test.experimental.sync_devices()
    except RuntimeError:
        # If something runs in graph mode, this will complain.
        # In your benchmark path you want eager execution anyway.
        pass

class Pipeline:
    """Main Pipeline class to process all positions."""

    def __init__(
        self,
        xpreader: utils.XPReader,
        config: Config,
        resfolder: Optional[Union[str, Path]] = None,
        gtfolder: Optional[Union[str, Path]] = None,
        reload: bool = False,
    ) -> None:
        """
        Initialize Pipeline.

        Parameters
        ----------
        xpreader : utils.XPReader
            utilities XPReader object.
        config : Config
            DeLTA configuration.
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
        self.reader: utils.XPReader = xpreader
        "Experiment reader object"
        self.positions: Dict[int, Position] = {}
        "Dict of Position objects for experiment"
        self.config: Config = config
        "Configuration parameters object"
        self.resfolder: Path
        "Folder to save results to"
        self.gtfolder: Path
        "Folder containing ground truth for evaluation"

        if resfolder is None:
            xpfile = self.reader.path
            assert xpfile is not None
            if self.reader.filetype == "dir":
                resfolder = self.reader.filehandle / "delta_results"
            else:
                resfolder = xpfile.with_name(xpfile.stem + "_delta_results")

        self.resfolder = Path(resfolder)
        self.gtfolder = Path(gtfolder) if gtfolder is not None else None

        self.resfolder.mkdir(exist_ok=True)

        # Initialize position processors:
        for iposition in self.reader.positions:
            self.positions[iposition] = Position(
                iposition,
                config=self.config,
            )

        # If reload flag, reload positions from pickle files:
        if reload:
            for iposition in self.reader.positions:
                self.positions[iposition] = Position.load(
                    self.resfolder / f"Position{iposition:06d}.pkl"
                )

    def process(
        self,
        positions: Optional[List[int]] = None,
        frames: Optional[Union[range, slice]] = None,
        clear: bool = True,
    ) -> None:
        """
        Run pipeline.

        Parameters
        ----------
        positions : list of int or None, optional
            List of positions to run. If None, all positions are run.
            The default is None.
        frames : int or None, optional
            Number of frames to run. If None, all frames are run.
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
            frames = self.reader.frames
        elif isinstance(frames, slice):
            assert frames.step == 1
            if frames.start is None:
                frames = slice(self.reader.frames.start, frames.stop)
            if frames.stop is None:
                frames = slice(frames.start, self.reader.frames.stop)
            frames = range(frames.start, frames.stop)

        if positions is None:
            positions = self.reader.positions

        # Run through positions
        for iposition, position in self.positions.items():
            if iposition not in positions:
                continue

            all_frames = self.reader.getframes(position=iposition, frames=frames)

            # Create ROIs and distribute images
            position.preprocess(all_frames, frames=frames)

            # Segment all ROIs
            sync_tf()
            seg_start = perf_counter()
            position.segment(frames=frames)
            sync_tf()
            self.seg_time = perf_counter() - seg_start

            # Track cells in all ROIs
            sync_tf()
            track_start = perf_counter()
            position.track(frames=frames)
            sync_tf()
            self.track_time = perf_counter() - track_start

            self.total_time = self.seg_time + self.track_time

            # Extract features in all ROIs
            position.features(frames=frames)

            # Save to disk and clear memory
            position.save(
                filename=self.resfolder / f"Position{iposition:06d}",
                frames=frames,
                reader=self.reader,
                save_format=self.config.save_format,
            )

            self._write_frame_metrics_csv(position, iposition, frames)

            if clear:
                position.clear()

    def _write_frame_metrics_csv(self, position, iposition, frames):
        import re

        seq_csv = self.resfolder / f"{iposition:02d}_frame_metrics.csv"
        write_header = (not seq_csv.exists()) or seq_csv.stat().st_size == 0

        def gt_cells_for_frame(frame: int) -> int:
            if self.gtfolder is None:
                return -1

            for prefix in ("man_seg", "man_track"):
                gt_file = self.gtfolder / f"{prefix}{frame:03d}.tif"
                if gt_file.exists():
                    gt_img = cv2.imread(str(gt_file), cv2.IMREAD_UNCHANGED)
                    if gt_img is None:
                        continue
                    labels = np.unique(np.squeeze(gt_img))
                    return int(np.count_nonzero(labels > 0))

            return -1

        with seq_csv.open("w", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(
                    [
                        "sequence",
                        "frame",
                        "seconds",
                        "fps",
                        "gt_cells",
                        "detected_cells",
                        "seg_time_s",
                        "track_time_s",
                    ]
                )

            # Only write integer frame keys. Anything else is legacy/old-data noise.
            for frame in sorted(position.frame_metrics):
                if not isinstance(frame, int):
                    continue

                metrics = position.frame_metrics[frame]

                seg_time = metrics.get("seg_time", 0.0) or 0.0
                track_time = metrics.get("track_time", 0.0) or 0.0

                seconds = seg_time + track_time
                
                if seconds <= 0:
                    continue

                fps = 1.0 / seconds
                frame_label = f"t{frame:03d}"
                gt_cells = gt_cells_for_frame(frame)
                detected_cells = (
                    metrics["detected_cells"] if metrics["detected_cells"] is not None else ""
                )

                writer.writerow(
                    [
                        f"{iposition:02d}",
                        frame_label,
                        f"{seconds:.6f}",
                        f"{fps:.6f}",
                        gt_cells,
                        detected_cells,
                        f"{seg_time:.6f}",
                        f"{track_time:.6f}",
                    ]
                )


class Position:
    """Position-processing object."""

    def __init__(
        self,
        position_nb: int,
        config: Config,
    ) -> None:
        """
        Initialize Position.

        Parameters
        ----------
        position_nb : int
            Position index.
        config : Config
            DeLTA configuration.

        Returns
        -------
        None.

        """
        self.position_nb = position_nb
        "Position index number in the experiment"
        self.rois: List[ROI] = []
        "List of ROI objects under position"
        self.drift_values: Tuple = ([], [])
        "XY drift correction values over time"
        self.rotate: float = 0.0
        "Angle to rotate images by, in degrees"
        self.driftcorbox: CroppingBox
        "Bounding box of the reference for drift correction"
        self.drifttemplate: Optional[utils.Image]
        "Rotation angle of the position to compensate"
        self.config: Config = config
        "Configuration parameters object"
        self.verbose: int = 1
        "Console output verbosity"
        self.frame_metrics = defaultdict(
            lambda: {
                "seg_time": 0.0,
                "track_time": 0.0,
                "gt_cells": None,
                "detected_cells": None,
            }
        )
        "Dict for storing segmentation and tracking times for each frame"

    def __eq__(self, other) -> bool:
        """Equality function for Position."""
        eq_position_nb = self.position_nb == other.position_nb
        eq_rois = self.rois == other.rois
        eq_drift_values = np.array_equal(self.drift_values, other.drift_values)
        eq_rotate = self.rotate == other.rotate
        eq_verbose = self.verbose == other.verbose
        eqs = [
            eq_position_nb,
            eq_rois,
            eq_drift_values,
            eq_rotate,
            eq_verbose,
        ]
        return all(eqs)

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
        all_frames: npt.NDArray[np.float32],
        frames: range,
        reference: Optional[utils.Image] = None,
    ) -> None:
        """
        Pre-process position (Rotation correction, identify ROIs, initialize drift correction).

        Parameters
        ----------
        all_frames : npt.NDArray[np.float32]
            All frames for this position. Shape: (frames, channels, Y, X).
        frames : range
            Range of frames to preprocess.
        reference : 2D array, optional
            Reference image to use to perform pre-processing. If None,
            the first image of each position will be used.
            The default is None.

        Returns
        -------
        None.

        """
        self._msg("Starting pre-processing")

        # If no reference frame provided, read first frame from reader
        reference = np.array(all_frames[0, 0, :, :] if reference is None else reference)

        # Rotation correction
        if self.config.rotation_correction:
            self.rotate = utils.deskew(reference)
            self._msg(f"Rotation correction: {self.rotate} degrees")
            for iframe in range(all_frames.shape[0]):
                for ichannel in range(all_frames.shape[1]):
                    all_frames[iframe, ichannel, :, :] = utils.imrotate(
                        all_frames[iframe, ichannel, :, :], self.rotate
                    )
            reference = utils.imrotate(reference, self.rotate)

        # Find ROIs
        if "rois" in self.config.models:
            roi_boxes = Position.find_roi_boxes(reference, self.config)
        else:
            roi_boxes = [CroppingBox.full(reference)]

        # Get drift correction template and box
        if self.config.drift_correction:
            self.drifttemplate = utils.getDriftTemplate(
                roi_boxes,
                reference,
                whole_frame=self.config.whole_frame_drift,
            )
            self.driftcorbox = CroppingBox.full(reference)
            if not self.config.whole_frame_drift:
                self.driftcorbox.ybr = max(box.ytl for box in roi_boxes)

            # Correct drift:
            all_frames[:, 0, :, :], self.drift_values = utils.driftcorr(
                all_frames[:, 0, :, :],
                template=self.drifttemplate,
                box=self.driftcorbox,
            )
            for ichannel in range(1, all_frames.shape[1]):
                all_frames[:, ichannel, :, :], _ = utils.driftcorr(
                    all_frames[:, ichannel, :, :], drift=self.drift_values
                )

        # Instantiate ROIs:
        self.rois = [
            ROI(
                img_stack=[box.crop(img) for img in all_frames[:, 0, :, :]],
                fluo_stack=[
                    [box.crop(img) for img in imgs] for imgs in all_frames[:, 1:, :, :]
                ],
                roi_nb=iroi,
                first_frame=frames[0],
                box=box,
                config=self.config,
            )
            for iroi, box in enumerate(roi_boxes)
        ]

    @staticmethod
    def find_roi_boxes(reference: utils.Image, config: Config) -> List[CroppingBox]:
        """
        Use U-Net to detect ROIs (chambers etc...).

        Parameters
        ----------
        reference : utils.Image
            Reference image to use to detect ROIs.
        config : Config
            DeLTA configuration object.

        Returns
        -------
        boxes : List[CroppingBox]
            List of ROI boxes.

        """
        # Rescale pixel values between 0 and 1 for the old model
        reference = (reference - reference.min()) / reference.ptp()

        # Predict
        rois_pred = config.model("rois").predict(
            cv2.resize(reference, config.target_size_rois)[
                np.newaxis, :, :, np.newaxis
            ],
            verbose=0,
        )

        # Clean up
        rois_mask = postprocess(
            cv2.resize(np.squeeze(rois_pred), reference.shape[::-1]),
            min_size=config.min_roi_area,
        )

        # Get boxes
        # Implementation note: cv2.findContours (even including
        # cv2.boundingRect) is about twice as fast as
        # cv2.connectedComponentsWithStats here.
        roi_boxes = []
        contours = utils.find_contours(rois_mask)
        for chamber in contours:
            xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
            roi_boxes.append(
                CroppingBox(
                    xtl=xtl,
                    # -10% of height to make sure the top is not cropped
                    ytl=ytl - int(0.1 * boxheight),
                    xbr=xtl + boxwidth,
                    ybr=ytl + boxheight,
                )
            )

        # Sorting by top-left X (normally sorted by top-left Y)
        roi_boxes.sort(key=lambda box: box.xtl)

        return roi_boxes

    def segment(self, frames: range) -> None:
        """
        Segment cells in all ROIs in position.

        Parameters
        ----------
        frames : range
            Frames to run.

        Returns
        -------
        None.

        """
        self._msg(f"Starting segmentation ({len(frames)} frames)")

        segmentation_model = self.config.model("seg")

        for roi in self.rois:
            roi.position = self

        for iroi, roi in enumerate(self.rois, start=1):
            self._msg(f"Segmentation - ROI {iroi}/{len(self.rois)}")
            roi.segment(frames, segmentation_model)
            
        

    def track(self, frames: range) -> None:
        """
        Track cells in all ROIs in position.

        Parameters
        ----------
        frames : range
            Frames to track.

        Returns
        -------
        None.

        """
        self._msg(f"Starting tracking ({len(frames)} frames)")

        tracking_model = self.config.model("track")

        for roi in self.rois:
            roi.position = self

        for iroi, roi in enumerate(self.rois, start=1):
            self._msg(f"Tracking - ROI {iroi}/{len(self.rois)}")
            roi.track(frames, tracking_model)

    def features(self, frames: range) -> None:
        """
        Extract features for all ROIs in frames.

        Parameters
        ----------
        frames : range
            Frames to run.

        Returns
        -------
        None.

        """
        self._msg(f"Starting feature extraction ({len(frames)} frames)")

        # Now extracting multi-frame features
        for roi in self.rois:
            for cellid, cell in roi.lineage.cells.items():
                for iframe in cell["frames"]:
                    for feature in ["length", "area"]:
                        try:
                            gr = roi.lineage.growthrate(cellid, iframe, feature)
                            roi.lineage.setvalue(
                                cellid, iframe, f"growthrate_{feature}", gr
                            )
                        except KeyError:
                            warnings.warn(
                                "An issue happened related to feature extraction: "
                                "some cells cannot be found after resizing."
                            )

    def save(
        self,
        filename: Optional[Union[str, Path]] = None,
        frames: Optional[range] = None,
        reader: Optional[utils.XPReader] = None,
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
        frames : Optional[range]
            Range of frames to save. If None, all frames will be saved.
            The default is None.
        reader : Optional[utils.XPReader]
            Only needed if saving as a movie.
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
            if reader is None:
                raise ValueError("If saving as matlab, xpreader cannot be None.")
            data = self.legacysave(reader)
            savemat(filename.with_suffix(".mat"), data)

        if "pickle" in save_format:
            self._msg("Saving to pickle format\n" + str(filename.with_suffix(".pkl")))
            with open(filename.with_suffix(".pkl"), "wb") as file:
                pickle.dump(self, file)

        if "movie" in save_format:
            self._msg("Saving results movie\n" + str(filename.with_suffix(".mp4")))
            if reader is None:
                raise ValueError("If saving as movie, xpreader cannot be None.")
            movie = self.results_movie(frames, reader)
            utils.vidwrite(movie, (filename.parent / 'movie' / filename.with_suffix(".mp4").name), verbose=False)

        if "CTC" in save_format:
            self._msg("Saving ctc results\n" + str(filename.with_suffix(".txt")))
            if reader is None:
                raise ValueError("If saving results in ctc format, xpreader cannot be None.")
            self.save_results_ctc(frames, filename)

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
            Reloaded Position object. (without reader)

        """
        with open(filename, "rb") as file:
            position = pickle.load(file)

        return position

    def clear(self) -> None:
        """
        Clear Position-specific variables from memory (can be loaded back with load()).

        Returns
        -------
        None.

        """
        self._msg("Clearing variables from memory")
        for k in self.__dict__:
            setattr(self, k, None)

    def legacysave(self, reader: utils.XPReader) -> Dict[str, Any]:
        """
        Save pipeline data in the legacy Matlab format.

        Parameters
        ----------
        res_file : str or Path
            Path to save file.
        reader : utils.XPReader
            Experiment reader.

        Returns
        -------
        None.

        """
        # File reader info
        moviedimensions = [
            reader.y,
            reader.x,
            len(reader.channels),
            len(reader.frames),
        ]
        xpfile = reader.path

        # If No ROIs detected for position:
        if len(self.rois) == 0:
            return {
                "res": [],
                "tiffile": str(xpfile),
                "moviedimensions": moviedimensions,
                "proc": {"rotation": self.rotate, "chambers": [], "XYdrift": []},
            }

        # Initialize data structure/dict:
        data: Dict[str, Any] = {
            "moviedimensions": moviedimensions,
            "tifffile": str(xpfile),
        }

        # Proc dict/structure:
        data["proc"] = {
            "rotation": self.rotate,
            "XYdrift": np.array(self.drift_values, dtype=np.float32),
            "chambers": np.array(
                [
                    [
                        r.box.xtl,
                        r.box.ytl,
                        r.box.xbr - r.box.xtl,
                        r.box.ybr - r.box.ytl,
                    ]
                    for r in self.rois
                ],
                dtype=np.float32,
            ),
        }

        # Lineages:
        data["res"] = []
        for r in self.rois:
            res: Dict[str, Any] = {}

            # Resized labels stack: (ie original ROI size)
            res["labelsstack_resized"] = np.array(r.label_stack, dtype=np.uint16)

            # Not resized stack: (ie U-Net seg target size)
            label_stack = []
            for frame, cellnbs in r.lineage.cellnumbers.items():
                label_stack += [
                    utils.label_seg(r.get_seg(frame), [c + 1 for c in cellnbs])
                ]
            res["labelsstack"] = np.array(label_stack, dtype=np.uint16)

            # Run through cells, update to 1-based indexing
            lin: List[Dict[str, Any]] = []
            for cell in r.lineage.cells.values():
                lin += [{}]
                # Base lineage:
                lin[-1]["mother"] = (
                    cell["mother"] + 1 if cell["mother"] is not None else 0
                )
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
                if "fluo" in cell:
                    lin[-1]["fluo"] = np.array(cell["fluo"])
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

        return data

    def save_results_ctc(self,frames,filename):
                
        ctc_folder = filename.parent / 'CTC' / f'{self.position_nb:02d}'
        ctc_folder.mkdir(exist_ok=True)

        assert len(self.rois) == 1

        roi = self.rois[0]
        labels_stack = roi.label_stack
        lineage = roi.lineage
        ctc_data = None

        for fidx, label_stack in zip(frames,labels_stack):
            cellnbs = np.unique(label_stack)
            cellnbs = cellnbs[cellnbs > 0] - 1

            mask = np.copy(label_stack)

            if ctc_data is None:
                if len(cellnbs) > 0:
                    ctc_data = []
                    for cell in cellnbs:
                        ctc_data.append(np.array([cell+1,fidx,fidx,0]))
                    ctc_data = np.stack(ctc_data)
                    
                ctc_cells = cellnbs + 1
                prevcellnbs = np.copy(cellnbs)

            else:
                max_cellnb = ctc_data.shape[0]

                ctc_cells_new = np.ones_like(cellnbs) * -1
                mask_copy = np.copy(mask)
                daughters = []
                for c,prevcellnb in enumerate(prevcellnbs):
                    if prevcellnb in cellnbs:
                        ctc_cell = ctc_cells[c]
                        prevfidx_ind = lineage.cells[prevcellnb]['frames'].index(fidx-1)
                        fidx_ind = prevfidx_ind + 1

                        if fidx_ind < len(lineage.cells[prevcellnb]['frames']) and lineage.cells[prevcellnb]['daughters'][fidx_ind] is not None:
                            dau = lineage.cells[prevcellnb]['daughters'][fidx_ind]
                            assert dau in cellnbs
                            daughters.append(dau)
                            max_cellnb += 1
                            new_cell_1 = np.array([max_cellnb,fidx,fidx,ctc_cell])[None]
                            ctc_cells_new[cellnbs == dau] = max_cellnb 
                            mask[mask_copy == dau+1] = max_cellnb
                            max_cellnb += 1
                            new_cell_2 = np.array([max_cellnb,fidx,fidx,ctc_cell])[None]
                            ctc_data = np.concatenate((ctc_data,new_cell_1,new_cell_2),axis=0)
                            ctc_cells_new[cellnbs == prevcellnb] = max_cellnb
                            mask[mask_copy == prevcellnb+1] = max_cellnb
                                
                        else:
                            ctc_data[ctc_cell-1,2] = fidx
                            ctc_cells_new[cellnbs == prevcellnb] = ctc_cell
                            mask[mask_copy == prevcellnb+1] = ctc_cell
                
                for c,cellnb in enumerate(cellnbs):
                    if cellnb not in prevcellnbs and cellnb not in daughters:
                        max_cellnb += 1
                        new_cell = np.array([max_cellnb,fidx,fidx,0])[None]
                        ctc_data = np.concatenate((ctc_data,new_cell),axis=0)

                        ctc_cells_new[cellnbs == cellnb] = max_cellnb     
                        mask[mask_copy == cellnb+1] = max_cellnb

                assert -1 not in ctc_cells_new

                ctc_cells = np.copy(ctc_cells_new)
                prevcellnbs = np.copy(cellnbs)

            cv2.imwrite(str(ctc_folder / f'mask{fidx:03d}.tif'),mask)

        np.savetxt(ctc_folder / 'res_track.txt',ctc_data,fmt='%d')


    def results_movie(self, frames: Optional[range], reader: utils.XPReader) -> Any:
        """
        Generate movie illustrating segmentation and tracking.

        Returns
        -------
        movie : list of 3D numpy arrays
            List of compiled movie frames
        frames : Optional[range]
            Range of frames to use for the movie.  If None, all of them will be used.
            The default is None.
        reader : utils.XPReader
        """
        # Re-read trans frames:
        trans_frames = reader.getframes(
            position=self.position_nb,
            channels=reader.channels[0],
            frames=frames,
            rotate=self.rotate,
        )
        if frames is None:
            frames = reader.frames
        trans_frames = trans_frames[:, 0]
        trans_frames = trans_frames * (1 / trans_frames.max())

        if self.config.drift_correction:
            trans_frames, _ = utils.driftcorr(trans_frames, drift=self.drift_values)
        movie = []

        # Run through frames, compile movie:
        for iframe, frame in zip(frames, trans_frames):
            # RGB-ify:
            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)

            # Add frame number text:
            frame = cv2.putText(
                frame,
                text=f"frame {iframe:06d}",
                org=(int(frame.shape[0] * 0.05), int(frame.shape[0] * 0.97)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 1, 1, 1),
                thickness=2,
            )

            for r, roi in enumerate(self.rois):
                # Get chamber-specific variables:
                colors = utils.getrandomcolors(list(roi.lineage.cells.keys()), seed=r)
                fr = roi.get_labels(iframe)
                assert fr is not None  # FIXME: why is it not None?
                cells, contours = utils.getcellsinframe(fr, return_contours=True)

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
                    oldpole = roi.lineage.getvalue(cell, iframe, "old_pole")
                    assert isinstance(oldpole, np.ndarray)  # for mypy
                    frame = cv2.drawMarker(
                        frame,
                        (oldpole[1] + xtl, oldpole[0] + ytl),
                        color=colors[cell],
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=3,
                        thickness=1,
                    )

                    daughter = roi.lineage.getvalue(cell, iframe, "daughters")
                    bornago = roi.lineage.cells[cell]["frames"].index(iframe)
                    mother = roi.lineage.cells[cell]["mother"]

                    if daughter is None and (bornago > 0 or mother is None):
                        newpole = roi.lineage.getvalue(cell, iframe, "new_pole")
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
                        newpole = roi.lineage.getvalue(cell, iframe, "new_pole")
                        daupole = roi.lineage.getvalue(daughter, iframe, "new_pole")
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
    """ROI processor object."""

    def __init__(
        self,
        img_stack: List[utils.Image],
        fluo_stack: List[List[utils.Image]],
        roi_nb: int,
        first_frame: int,
        box: CroppingBox,
        config: Config,
    ) -> None:
        """
        Initialize ROI.

        Parameters
        ----------
        img_stack : List[utils.Image]
            Image stack.
        fluo_stack : List[List[utils.Image]]
            Fluo stack. Shape: (frames, fluo_channels, Y, X).
        roi_nb : int
            ROI index.
        first_frame : int
            Index of the first frame (in general 0 or 1).
        box : CroppingBox
            CroppingBox for ROI.
        config : Config
            DeLTA configuration.

        Returns
        -------
        None.

        """
        self.roi_nb = roi_nb
        "The ROI index number"
        self.box = box
        "ROI crop box"
        self.first_frame = first_frame
        "Index of the first frame"
        self.img_stack: List[utils.Image] = [
            (img - img.min()) / img.ptp() for img in img_stack
        ]
        "Input images stack"
        self.fluo_stack = np.array(fluo_stack)
        if self.fluo_stack.ndim == 2:
            # No fluo channels
            self.fluo_stack = self.fluo_stack[:, :, np.newaxis, np.newaxis]
        "Fluo images stack"
        self.seg_stack: List[utils.SegmentationMask] = []
        "Segmentation images stack"
        self.lineage = utils.Lineage()
        "Lineage object for ROI"
        self.label_stack: List[utils.Labels] = []
        "Labelled images stack"
        self.config: Config = config
        "Configuration parameters object"
        self.scaling: Optional[Tuple[float, float]]
        "Resizing ratios along Y and X"

        if config.crop_windows:
            self.scaling = None
        else:
            self.scaling = (
                (box.ybr - box.ytl) / self.config.target_size_seg[0],
                (box.xbr - box.xtl) / self.config.target_size_seg[1],
            )

    def __eq__(self, other) -> bool:
        """Equality function for ROI."""
        print(f"ROI {self.roi_nb}")
        eq_roi_nb = self.roi_nb == other.roi_nb
        if not eq_roi_nb:
            print("ROIs differ by roi_nb")
        eq_box = self.box == other.box
        if not eq_box:
            print("ROIs differ by box")
            print(self.box)
            print(other.box)
        eq_first_frame = self.first_frame == other.__dict__.get("first_frame", 0)
        if not eq_first_frame:
            print("ROIs differ by first_frame")
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
            eq_first_frame,
            eq_img_stack,
            eq_seg_stack,
            eq_label_stack,
        ]
        return all(eqs)

    def get_img(self, frame: int) -> utils.Image:
        """
        Return the ROI image at a given frame.

        Parameters
        ----------
        frame : int
            Frame index.

        Returns
        -------
        image : utils.Image
            Image of the ROI.
        """
        return self.img_stack[frame - self.first_frame]

    def get_fluo(self, frame: int) -> utils.Image:
        """
        Return the ROI fluo images at a given frame.

        Parameters
        ----------
        frame : int
            Frame index.

        Returns
        -------
        image : utils.Image
            List of fluo images of the ROI.
        """
        return self.fluo_stack[frame - self.first_frame]

    def get_seg(self, frame: int) -> utils.SegmentationMask:
        """
        Return the ROI segmentation mask at a given frame.

        Parameters
        ----------
        frame : int
            Frame index.

        Returns
        -------
        seg : utils.SegmentationMask
            Segmentation mask of the ROI.
        """
        return self.seg_stack[frame - self.first_frame]

    def get_labels(self, frame: int) -> utils.Labels:
        """
        Return the ROI labels at a given frame.

        Parameters
        ----------
        frame : int
            Frame index.

        Returns
        -------
        labels : utils.Labels
            Labels frame.
        """
        return self.label_stack[frame - self.first_frame]

    def segment(self, frames: range, model: Optional[tf.keras.Model] = None) -> None:
        # Run through frames and compile segmentation inputs
        imgs = []
        windowss = []
        frame_keys = []
        n_windows_per_frame = []

        for frame in frames:
            inputs, windows = self.get_segmentation_inputs(frame)
            imgs.append(inputs)
            windowss.append(windows)
            n_windows_per_frame.append(len(inputs))

            # Use a stable per-frame key here
            frame_keys.append(f"t{frame:03d}" if isinstance(frame, int) else str(frame))

        inputs = np.concatenate(imgs)

        # Run segmentation model
        sync_tf()
        t0 = perf_counter()
        logits = model.predict(inputs, batch_size=self.config.pipeline_seg_batch)
        sync_tf()
        elapsed = perf_counter() - t0

        # Split the batch output back per frame 
        offset = 0
        total_windows = sum(n_windows_per_frame)

        for frame_key, frame, windows, nwin in zip(frame_keys, frames, windowss, n_windows_per_frame):
            frame_logits = logits[offset : offset + nwin]
            offset += nwin

            self.process_segmentation_outputs(
                frame_logits,
                frame=frame,
                windows=windows,
            )

            # Store per-frame segmentation time.
            # This splits the batch time proportionally by number of windows.
            self.position.frame_metrics[frame]["seg_time"] += elapsed * (nwin / total_windows)

    def get_segmentation_inputs(
        self, frame: int
    ) -> Tuple[np.ndarray, Optional[Tuple[List, List]]]:
        """
        Compile segmentation inputs for ROI.

        Parameters
        ----------
        frame : int
            Frame number for the segmentation inputs.

        Returns
        -------
        x : 4D array
            Segmentation input array. Dimensions are
            (windows, *self.config.target_size_seg, 1).
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        """
        # Crop and scale:
        i = self.get_img(frame)

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
        logits: npt.NDArray[np.float32],
        frame: int,
        windows: Optional[Tuple[List, List]] = None,
    ) -> None:
        """
        Process outputs after they have been segmented.

        Parameters
        ----------
        logits : 4D array
            Segmentation output array. Dimensions are
            (windows, *self.config.target_size_seg, 1).
        frame : int
            Frame index.
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        Returns
        -------
        None.

        """
        # Stitch windows back together (if needed):
        if windows is None:
            logits = logits[0, :, :, 0]
        else:
            logits = utils.stitch_pic(logits[..., 0], windows[0], windows[1])

        # Binarize:
        seg = binarizerange(logits, threshold=0).astype(np.uint8)
        # Crop out segmentation if image was smaller than target_size
        if self.config.crop_windows:
            seg = seg[: self.img_stack[0].shape[0], : self.img_stack[0].shape[1]]
        # Area filtering:
        seg = utils.opencv_areafilt(seg, min_area=self.config.min_cell_area)

        n_components, _ = cv2.connectedComponents(seg.astype(np.uint8))
        detected_cells = int(max(0, n_components - 1))

        self.position.frame_metrics[frame]["detected_cells"] = detected_cells

        # Append to segmentation results stack:
        assert len(self.seg_stack) == frame - self.first_frame
        self.seg_stack.append(seg)

    def track(self, frames: range, model: Optional[tf.keras.Model] = None) -> None:
        # compile inputs and remember per-frame counts
        inpts = []
        references = []
        n_cells_per_frame = []

        for frame in frames:
            inputs, boxes = self.get_tracking_inputs(frame=frame)
            inpts.append(inputs)
            references.append((len(inputs), boxes))
            n_cells_per_frame.append(len(inputs))

        inputs = np.concatenate(inpts) if inpts else np.empty((0, *self.config.target_size_track, 1))

        sync_tf()
        t0 = perf_counter()
        if inputs.shape[0] > 0:
            logits = model.predict(inputs, batch_size=self.config.pipeline_track_batch)
        else:
            logits = np.empty(shape=(0, *self.config.target_size_track, 1))
        sync_tf()
        elapsed = perf_counter() - t0

        total_cells = sum(n_cells_per_frame) or 1

        i = 0
        for frame, (nbcells, boxes) in zip(frames, references):
            self.process_tracking_outputs(
                logits[i : i + nbcells],
                frame=frame,
                boxes=boxes,
            )
            i += nbcells

            self.position.frame_metrics[frame]["track_time"] += elapsed * (nbcells / total_cells)

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
        inputs : 4D array or None
            Tracking input array. Dimensions are
            (previous_cells, *self.config.target_size_track, 4).
        boxes : List of tuples of 2 dicts
            Crop and fill boxes to re-place outputs in the ROI.

        """
        # Check if segmentation data is ready:
        if len(self.seg_stack) <= frame - self.first_frame:
            raise RuntimeError(f"Segmentation incomplete - frame {frame}")

        # Get cell contours from previous frame
        if frame == self.first_frame:
            prev_cell_contours = []
        else:
            prev_cell_contours = utils.find_contours(self.get_seg(frame - 1))

        # Sort them along Y
        prev_cell_contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))

        # Allocate empty tracking inputs array:
        inputs = np.empty(
            (len(prev_cell_contours), *self.config.target_size_track, 4),
            dtype=np.float32,
        )

        # Get current and previous image, used as tracking inputs and resize them if needed
        curr_img = self.get_img(frame)
        prev_img = self.get_img(frame - 1)
        if not self.config.crop_windows:
            curr_img = cv2.resize(curr_img, dsize=self.config.target_size_seg[::-1])
            prev_img = cv2.resize(prev_img, dsize=self.config.target_size_seg[::-1])
            cb, fb = CroppingBox.full(curr_img), CroppingBox.full(curr_img)
            draw_offset = None
            boxes = [(cb, fb)] * len(prev_cell_contours)
        else:
            shape = cast(Tuple[int, int], curr_img.shape)  # for mypy
            # Cell-centered crop boxes
            boxes = [
                utils.gettrackingboxes(contour, shape, self.config.target_size_track)
                for contour in prev_cell_contours
            ]

        # Run through contours and compile inputs:
        for icontour, (contour, (cb, fb)) in enumerate(zip(prev_cell_contours, boxes)):
            if self.config.crop_windows:
                draw_offset = (-cb.xtl + fb.xtl, -cb.ytl + fb.ytl)

            # Current image
            fb.patch(inputs[icontour, :, :, 0], cb.crop(curr_img))

            # Segmentation mask of one previous cell (seed)
            inputs[icontour, :, :, 1] = cv2.drawContours(
                np.zeros(self.config.target_size_track, dtype=np.float32),
                [contour],
                0,
                offset=draw_offset,
                color=1.0,
                thickness=cv2.FILLED,
            )

            # Previous image
            fb.patch(inputs[icontour, :, :, 2], cb.crop(prev_img))

            # Segmentation of all current cells
            fb.patch(inputs[icontour, :, :, 3], cb.crop(self.get_seg(frame)))

        # Return tracking inputs and crop and fill boxes:
        return inputs, boxes

    def process_tracking_outputs(
        self,
        logits: npt.NDArray[np.float32],
        frame: int,
        boxes: List[Tuple[CroppingBox, CroppingBox]],
    ) -> None:
        """
        Process output from tracking U-Net.

        Get poles, update lineage and create label_stack.

        Parameters
        ----------
        logits : 4D array
            Tracking output array. Dimensions are
            (previous_cells, *self.config.target_size_track, 1).
        frame : int
            The frame to process for.
        boxes : List of tuples of 2 dicts
            Crop and fill boxes to re-place outputs in the ROI.

        Returns
        -------
        None.

        """
        # Get scores and attributions:
        # Label frame but numbered 1, 2, 3, 4, etc. (temporary labels)
        labels = utils.label_seg(self.get_seg(frame))
        scores = utils.getTrackingScores(labels, logits[:, :, :, 0], boxes=boxes)

        attributions = utils.getAttributions(scores)

        # Get poles:
        poles = utils.getpoles(self.get_seg(frame), labels, scaling=self.scaling)

        # Resize labels if not using crop windows:
        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)

        # Extract features for all cells in the ROI:
        extracted_features = utils.roi_features(
            labels,
            fluo_frames=self.get_fluo(frame),
        )

        # Make sure the same cell_ids are present in both dicts
        assert poles.keys() == extracted_features.keys()

        # Assign poles to extracted features:
        for cellid, (old_pole, new_pole) in poles.items():
            extracted_features[cellid].old_pole = old_pole
            extracted_features[cellid].new_pole = new_pole

        # Update lineage:
        self.lineage.update(None, frame)
        # Go through old cells:
        for o in range(attributions.shape[0]):
            attrib = attributions[o, :].nonzero()[0]
            features = [extracted_features[n] for n in attrib]
            self.lineage.update(o, frame, attrib=attrib, features=features)
        # Go through "orphan" cells:
        for n in range(attributions.shape[1]):
            attrib = attributions[:, n].nonzero()[0]
            features = [extracted_features[n]]
            if len(attrib) == 0:
                self.lineage.update(None, frame, attrib=[n], features=features)

        # Recompile label frame with new labels
        if frame not in self.lineage.cellnumbers:
            cell_nbs = []
        else:
            cell_nbs = [c + 1 for c in self.lineage.cellnumbers[frame]]
        labels = utils.label_seg(self.get_seg(frame), cell_nbs)

        # Resize image:
        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)

        assert len(self.label_stack) == frame - self.first_frame
        self.label_stack.append(labels)
