# DeLTA
[![Documentation Status](https://readthedocs.org/projects/delta/badge/?version=latest)](https://delta.readthedocs.io/en/latest/?badge=latest)
      
DeLTA (Deep Learning for Time-lapse Analysis) is a deep learning-based image processing pipeline for segmenting and tracking single cells in time-lapse microscopy movies.

See the manuscript on bioRxiv: [https://www.biorxiv.org/content/10.1101/720615v1](https://www.biorxiv.org/content/10.1101/720615v1)

##### Overview
Our pipeline is centered around two [U-Net](https://arxiv.org/abs/1505.04597) neural networks that are used sequentially:
1. To perform semantic binary segmentation of our cells as in the original U-Net paper.
2. To track cells from one movie frame to the next, and to identify cell divisions and mother/daughter cells.

In our application, we focus on *E. coli* cells trapped in a microfluidic device known as a ["mother machine"](https://www.cell.com/current-biology/fulltext/S0960-9822%2810%2900524-5). However, as shown in the manuscript and in the "yeast" branch of this repository, the pipeline can easily be adapted to other setups and organisms.

The U-Nets are implemented in Tensorflow 2.0 via the Keras API. We also provide a suite of scripts and GUIs in Matlab to interface DeLTA with Matlab and help users create their own training sets. See our [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing) repository for that.

Finally we provide datasets for training and evaluation, both for E. coli and yeast cells:
[DeLTA_data (google drive, 12GB zip file)](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP) [Note that unzipping this archive takes a long time and fails on some linux systems.]

## Install
### GPU Requirements
While this code can run on CPU, we recommend running it on GPU for faster processing. To do so you will need [an nvidia gpu with a cuda compute capability of 3.5 or higher](https://developer.nvidia.com/cuda-gpus). Make sure you update your gpu drivers ([from the nvidia website](https://www.nvidia.com/Download/index.aspx?lang=en-us))

### Tensorflow 2.0 - GPU (recommended)
We recommend installing tensorflow 2.0 (GPU)  with [anaconda](https://www.anaconda.com/distribution/). Anaconda is easy to install, and will allow you to setup environments that are not in conflict with each other and will streamline the installation process. See [here](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) for more details.

    (base)$ conda create --name tf_gpu python=3.6 tensorflow-gpu
The code should work with python >= 3.6, but just to be on the safe side we recommend you use the python=3.6 option when creating the tf_gpu environment. You can name your environment tf_gpu or whatever you like.

You will also need to install scikit-image in the environment:

    (base)$ conda activate tf_gpu
	(tf_gpu)$ conda install -c conda-forge scikit-image

And that's it! Although our code actually uses the Keras framework, since it is now included with tensorflow 2.0 as an API, you don't need to actually install it.

### Tensorflow 2.0 - CPU
If you can not install the GPU version of tensorflow (unfortunately the AMD graphics cards in Macs are not supported), you can install the CPU version and the code will run just fine. It will be significantly slower (probably by one to two orders of magnitude).

    (base)$ conda create --name tf_cpu python=3.6 tensorflow
    (base)$ conda activate tf_cpu
	(tf_cpu)$ conda install -c conda-forge scikit-image
	
Again you can name your environment whatever you like. 
	
### Elastic deformations (optional)
If you want to use random elastic deformations as described in [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597) (necessary for the yeast branch of this repository) you will need to install the relevant library from github:

    (base)$ conda activate tf_gpu
	(tf_gpu)$ conda install pip git
	(tf_gpu)$ pip install git+https://github.com/gvtulder/elasticdeform
	
Note: On Windows, you might need to [install c++ support for Visual Studio](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation)

## Example datasets
You can create your own datasets for your setup using our [suite of Matlab scripts and GUIs](https://gitlab.com/dunloplab/delta-interfacing), however we recommend you check out how data is formatted in our example datasets:
[DeLTA_data (google drive, 12GB zip file)](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP) [Note that unzipping this archive takes a long time and fails on some linux systems.]

Once unzipped, you should update the DeLTA_data folder path on line 6 of `train_seg.py`, `train_track.py`, `segmentation.py` and `tracking.py`.

## Example run:

The typical workflow of DeLTA is:
0. Design new training sets for segmentation and tracking (see [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing))
1. Train the segmentation and tracking U-Net on the training sets
2. Run trained segmentation U-Net on new data
3. Run trained tracking U-Net on output of segmentation U-Net
4. Run post-processing on tracking output to compile into a user-friendly format (see [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing))


### Training
You can test training with the datasets we provide with the `train_seg.py` function for segmentation and the `train_track.py` function for tracking. You can run them from within an IDE like spyder or from the shell.

### Prediction
Once your two models are trained, you can run predictions on larger datasets using the `segmentation.py` and `tracking.py` scripts sequentially. 

Our dataset features a pre-processed evaluation movie, but our pre-existing models have been quite good at generalizing to new data. We encourage you to test them directly, for example by cropping a chamber image sequence (or several) with software like imageJ and to process them with those two scripts, even without retraining the models. You would need to name the image sequence following our nomenclature:

Position01_Chamber01_Frame\*\*\*\.png

## Troubleshooting
### OOM - Out of memory (GPU)
If you run the code on your GPU, you might run into memory problems. This is both linked to the batch size and the size of the images you are using. The batch size is fairly straightforward to change, just lower the value at the beginning of the `train_seg.py` or `train_track.py` files. Note that lower batch sizes may mean slower training convergence or lower performance overall. You might want to increase the number of epochs or steps per epoch to compensate.

The other solution would be to use a smaller image target size. However, if the original images are, say 512×512, downsizing it to 256×256 will certainly reduce the memory footprint, but it might cause some borders between cells in the binary masks to disappear. Instead, you would have to modify your images size upstream of U-Net to make sure that your training set does feature cell-to-cell borders in the segmentation masks.
