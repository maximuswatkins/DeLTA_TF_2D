# DeLTA
> **NOTE**
This is version 2 of the DeLTA pipeline. For version 1, please check out branch 'version1'

DeLTA (Deep Learning for Time-lapse Analysis) is a deep learning-based image processing pipeline for segmenting and tracking single cells in time-lapse microscopy movies.

![](https://gitlab.com/dunloplab/delta/-/raw/images/DeLTAexample.gif)

Version 2 preprint: [O’Connor, O. M., Alnahhas, R. N., Lugagne, J.-B., & Dunlop, M. J. (2021). DeLTA 2.0: A deep learning pipeline for quantifying single-cell spatial and temporal dynamics. _BioRxiv_, 2021.08.10.455795](https://www.biorxiv.org/content/10.1101/2021.08.10.455795v1)


Version 1 paper:
[Lugagne, J.-B., Lin, H., & Dunlop, M. J. (2020). DeLTA: Automated cell segmentation, tracking, and lineage reconstruction using deep learning. _PLOS Computational Biology_, 16(4), e1007673](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673).

:bug: If you encounter bugs or have questions about the software, please use [Gitlab's issue system](https://gitlab.com/dunloplab/delta/-/issues)
##### Overview

This pipeline can process movies of rod-shaped bacteria growing in 2D setups such as agarose pads, as well as movies of *E. coli* cells trapped in a microfluidic device known as a ["mother machine"](https://www.cell.com/current-biology/fulltext/S0960-9822%2810%2900524-5). An earlier version of this software that only processed mother machine movies can be found on branch ['version1'](https://gitlab.com/dunloplab/delta/-/tree/version1) of this repository.

Our pipeline is centered around two [U-Net](https://arxiv.org/abs/1505.04597) neural networks that are used sequentially:
1. To perform semantic binary segmentation of our cells as in the original U-Net paper.
2. To track cells from one movie frame to the next, and to identify cell divisions and mother/daughter cells.

A third U-Net can be used to identify regions of interest in the image before perfoming segmentation. This is used with mother machine movies to identify single chambers, but by default only 1 ROI covering the entire field-of-view is used in the 2D version.

The U-Nets are implemented in Tensorflow 2 via the Keras API. We are currently working on the implementation graphical interfaces to generate training sets and to analyze and correct delta outputs. In the previous version of this software, we provided rudimentary Matlab interfaces in our [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing) repository.

We try to comment/document and make the code accessible and easily adaptable. We provide small scripts and example datasets to illustrate how the code works(see below).

## Installation
We are using this code on several different Linux and Windows systems, and unless otherwise specified the instructions apply for both.

### GPU Requirements
While this code can run on CPU, we recommend running it on GPU for significantly faster processing. To do so you will need [an nvidia gpu with a cuda compute capability of 3.5 or higher](https://developer.nvidia.com/cuda-gpus) and [an up-to-date gpu driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)). If no compatible GPU/driver is found on the system, tensorflow will run on CPU.

For more information see [here](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) and [here](https://www.tensorflow.org/guide/gpu).

### Anaconda installation
We recommend installing necessary modules and libraries with [Anaconda](https://www.anaconda.com/distribution/). Anaconda is easy to install, and will allow you to setup environments that are not in conflict with each other and will streamline the installation process.

From inside the DeLTA folder, the following commands will install and activate the environment:
```bash
(base)$ conda env create -f delta_env.yml
(base)$ conda activate delta_env
```
Note that the library versions listed in the [delta_env.yml](delta_env.yml) file have been tested and work on Linux and Windows, but more recent versions should work as well. We use the Spyder IDE, but it can be removed from the environment file.

> **Prerequisites**  
On Windows, you will need [c++](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation) and [.NET](https://docs.microsoft.com/en-us/dotnet/framework/install/guide-for-developers) support for Visual studio, and a recent [Java JDK](https://www.oracle.com/java/technologies/javase-downloads.html) (≥8).  
> On Linux, you will need a recent JDK/OpenJDK and will need to create a JAVA_HOME environment variable pointing to it, for example on Ubuntu:  
> ```bash
> (base)$ export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
> ```

### Configuration
Our latest trained models can be found [here](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP)

Once downloaded, you will need to modify paths in the [config.py]() file to point to the trained models. Other paths can be left unchanged until you need to train your own models.

The [config.py]() file may be stored in your home directory under '~/.delta/config.py'. The [utilities.py]() file will automatically look for a [config.py]() file at this location

If the path exists, a config module will be uploaded in the [utilities.py]() file, otherwise the local [config.py]() will be loaded as a module as in the [utilities.py]()

Storing your [config.py]() file in your home directory will allow multiple people to use the same DeLTA package. Also, this config.py file would not be tracked through git.

The [config_defaults.py]() file is provided for reference. [config.py]() is not tracked by git so as not to overwrite changes.

To switch to the mother machine processing mode, simply change `presets` at the top of the file to 'mothermachine'. You can modify the presets or create your own presets in [config.py]()

### Re-installing the environment
To update to more recent versions of the libraries, we recommend completely re-installing the environment:
```bash
(base)$ conda env remove -n delta_env
(base)$ conda clean --all
(base)$ conda env create -f delta_env.yml
```
### Training and evaluation datasets
You can create your own datasets for your setup manually or using our [Matlab scripts and GUIs](https://gitlab.com/dunloplab/delta-interfacing), however we recommend you check out how data is formatted in our example datasets first:
[DeLTA_data (google drive, 19GB zip file)](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP) [~38 GB unzipped]

## How to use:

### Training
You can test training with the datasets we provide with the [train_seg.py](train_seg.py) script for segmentation, the [train_track.py](train_track.py) script for tracking, and the [train_rois.py](train_rois.py) script for ROIs identification.

### Prediction
To process an entire experiment, simply run the [pipeline.py]() file. The pipeline can process Bio-Formats compatible files (.nd2, .czi, .ome-tiff...) or image sequences saved in folders and subfolders. See the `xpreader` class in [utilities.py]() for more information on the second option.

You will be asked to input the type of file you want to process, and then to select the file/folder.

The data is saved by default under 2 formats, Matlab files, pickle files, and an .mp4 movie illustrating the results is also generated. The structure of the Matlab file is described in ['data_structure.txt'](). The pickle files allow for the Position, ROI and Lineage objects generated by the pipeline to be reloaded to memory for further analysis.

We also provide 2 simpler scripts, [segmentation.py](segmentation.py) and [tracking.py](tracking.py), that can be run sequentially to illustrate the core principles of our approach.

## Troubleshooting
### OOM - Out of memory (GPU)
On GPU, you might run into memory problems. This is both linked to the batch size and the size of the images. The batch size is straightforward to change, lower the value at the beginning of the [train_seg.py](train_seg.py) or [train_track.py](train_track.py) files. Note that lower batch sizes may mean slower training convergence or lower performance overall.

The other solution would be to use a smaller image target size. However if the original training images and masks are for example 512×512, downsizing them to 256×256 will reduce the memory footprint, but it might cause some borders between cells in the binary masks to disappear. Instead, training images should be resized upstream of DeLTA to make sure that your training set does feature cell-to-cell borders in the segmentation masks.

### cuDNN (or other libraries) not loading
We have run into OOM errors or some GPU-related libraries failing to load or initialize on laptops. See the "Limiting GPU memory growth" section on [this tensorflow help page](https://www.tensorflow.org/guide/gpu). Setting the `memory_growth_limit` parameter in [config.py](config.py) to a set value in MB (eg 1024, 2048...) should solve the issue.

### max() arg is an empty sequence / Some cells or ROIs are not detected
In [config.py](), there are 2 area filtering parameters: `min_roi_area` and `min_cell_area` that you can lower or set to 0 or None.

### NaN values in fluorescence, cell length, area etc... time-series after processing
This should only happen if `crop_windows` is set to False in [config.py](). When images are resized to fit the U-Net input and then sized back to their original size, a cell may be shrunk out of existence after the resizing ste. This typically happens in chambers that are right on the edge of the FOV and that are only partially visible, or when a focussing error causes the segmentation U-Net to identify very small cells. You can try to filter them out with the `min_cell_area` parameter in [config.py]() or  simply ignore NaN values in the lineage.
