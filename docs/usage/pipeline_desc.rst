
Pipeline
=================================


`Source <https://gitlab.com/dunloplab/delta/-/blob/master/delta/pipeline.py>`_ |
:doc:`API <../_generated/delta.pipeline>`

The pipeline is the core element of DeLTA: Once an
:ref:`XPreader <xpreader>` has been initialized on a bio-formats compatible
file or a sequence of images, it can be passed to the `Pipeline` object to
initialize it and then run it. Depending on which :doc:`configuration <config>`
was loaded, it will:

0. (optional) Perform an ROI (ie mother machine chambers) detection
   step, a rotation correction step and a drift correction step.
#. Segment images
#. Track segmented cells through time and reconstruct the lineage
#. Extract features such as cell length, cell fluorescence etc...
#. Save data to disk (see :doc:`outputs`)

Basic usage
------------

The most basic usage is::

    reader = delta.utilities.xpreader('/path/to/file/or/folder')
    processor = delta.pipeline.Pipeline(reader)
    processor.process()

This will process all frames, for all positions in the movie, and will extract
all features. The :doc:`output files <outputs>` will be saved under
`processor.resfolder`, which by default points to a new folder within or next
to the input folder/file. You can also specify where to save results during init::

    processor = delta.pipeline.Pipeline(
        reader,
        resfolder='/path/to/results/folder/'
        )

Or after init but before processing::

    processor.resfolder = '/path/to/another/folder/'

See also the :ref:`run pipeline script <run_pipeline>` and the :ref:`xpreader class <xpreader>`

Selectively process frames, positions, and features
---------------------------------------------------
You can also specify subsamples of the data to analyze::

    processor.process(frames=list(range(30))

(Note that the frames list must be contiguous and start from 0)

::

    processor.process(
        positions=[1, 3, 34],
        features=('length','edges','fluo2')
        )

::

    frames_per_pos = {
        0: list(range(100)),
        2: list(range(23)),
        12: [0,1,2,3,4,5,6,7]
        }

    for pos, frames in frames_per_pos.items():
        processor.process(
            positions=[pos],
            frames=frames,
            features=('area','perimeter','fluo1')
            )

Preprocessing
-------------
You can perform the preprocessing step before starting the rest of the process.
The main interest is that by doing so, you can actually pass initial reference
images to do the preprocessing on directly::

    processor.preprocess(references=numpy_array_of_reference_images)
    processor.process()

Otherwise, process will run the preprocessing step automatically on the first
frame of each position.

More details
------------------

The pipeline module uses 3 main classes of objects

* The higher level object is the :doc:`Pipeline class <../_generated/delta.pipeline.Pipeline>`.
  Typically only one is
  instantiated per analysis. Its main purpose is to create and initialize the
  `Position` class processor objects (under the `Pipeline.positions` list)
  and to provide a simple interface to process an entire multi-position experiment
* The :doc:`Position class <../_generated/delta.pipeline.Position>` class objects
  are used to process a single, specific position
  of the experiment. To process a single position, the user can run for example::

      pos = delta.pipeline.Position(
          position_nb=5,
          reader=reader,
          models=delta.utils.loadmodels(),
          drift_correction=delta.config.drift_correction,
          crop_windows=delta.config.crop_windows
          )
      pos.preprocess(rotation_correction=delta.config.rotation_correction)
      pos.segment(list(range(reader.timepoints)))
      pos.track(list(range(reader.timepoints)))
      pos.save('/path/to/file_without_ext',save_format=('pickle','legacy'))

  Each position will have one or more `ROI` class object under its `Position.rois`
  list. The way this works is that, for each frame, the `ROI` objects will crop
  out and prepare inputs and pass them to the `Position` object, which will
  run all ROI inputs at once before dispatching the outputs back to the `ROI`
  objects
* | The :doc:`ROI class <../_generated/delta.pipeline.ROI>` objects are
    dedicated to one region of interest in the field of view.
    They will focus on one area, as defined under `ROI.box`, and prepare U-Net
    inputs for each timepoint. These inputs are sent to their 'parent' `Position`
    object to be processed with the inputs of all other `ROIs` for the timepoint.
  | The prediction outputs are then sent back to the `ROIs` to be post-processed.
    For the '2D' version, the `Position` object only have 1 roi each that works
    on the entire FOV. For the mother machine version, the `Position` has as many
    ROIs as there are chambers in the FOV.
  | The `ROI` objects can not process an ROI on their own, they need to be under
    a `Position` object that will run the actual `model.predict()` calls.

.. _feat_extr:

Feature extraction
-------------------
At the end of the pipeline run, single-cell features over time are extracted.
These include morphological features:

* | Cell area: The area of the cell, in pixels, as returned by opencv's
    contourArea(). This means that corner pixels are counted as 1/4 and
    straight edge pixels are coutned as 1/2.
* | Cell edges: The edges of the image that cell is currently touching. Left,
    right, bottom, and top edges are labelled as '-x', '+x', '+y', and '-y',
    respectively.
* | Cell length: The cell length, computed by fitting a rotated bounding box
    to the segmented cell. While this technique is fast, it is not as
	accurate for bent or filamented cells.
* | Position of the old pole: The position of the old pole of the cell in the
    image. The position is given as (Y,X) coordinates, with (0,0) in the
    top left corner	of the image (ie row-major ordering).
* | Position of the new pole: The position of the new pole of the cell in the
    image. The position is given as (Y,X) coordinates, with (0,0) in the
    top left corner	of the image (ie row-major ordering).
* | Cell perimeter: The number of pixels of the cell's contour.
* | Cell width: The cell width, computed by fitting a rotated bounding box
    to the segmented cell. While this technique is fast, it is not as
	accurate for bent or filamented cells.

And dynamical features, such as growth rates:

* | Growth rate (area-based): The instantaneous exponential rate of increase
    of the cell area is extracted, with centered differences when the cell
    exists at both previous and next time points, and one-sided differences
    otherwise.
* | Growth rate (length-based): The instantaneous exponential rate of increase
    of the cell length is extracted, with centered differences when the cell
    exists at both previous and next time points, and one-sided differences
    otherwise.

Using central differences allows the error to decrease quadratically with the
time interval between frames.  In both cases, the growth rate computation
should behave well even during cell divisions, but tracking mistakes can affect
it negatively. Besides, imaging and segmentation noise can produce a non-smooth
growth rate which one might find suitable to smooth with an appropriate function
(such as, for example, centered moving averages).

If fluorescence channels are also provided, the pipeline will also extract
the average fluorescence for each channel. The mean intensity over all pixels
of the segmented cell's surface is computed.

See also :doc:`outputs` and :doc:`analysis`

Reloading objects
---------------------

After results have been saved to disk in the pickle format, they can be
reloaded with the following commands::

    # Init a new Pipeline:
    proc = delta.pipeline.Pipeline(reader)
    # Reload position data:
    proc.positions[0].reload('/path/to/savefile0.pkl')
    proc.positions[3].reload('/path/to/savefile3.pkl')


For more information on the properties of the `Position` and `ROI` objects, and
how to use them instead of the data in the MAT files, see :doc:`outputs`
