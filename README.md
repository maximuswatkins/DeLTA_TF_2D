# U-Net MoMa
This repository contains an implementation of the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) neural network in tensorflow 2.0 (through the Keras API)

We use it to analyze timelapse microscopy movies of bacteria. In our application, the cells are trapped in a microfluidic device known as a ["mother machine"](https://www.cell.com/current-biology/fulltext/S0960-9822%2810%2900524-5). However, the approach should be applicable to other setups or organisms.

This implementation of U-Net is based on [Zhixuhao's](https://github.com/zhixuhao/unet) but has been significantly modified. We used it to achieve two goals:
1. To perform semantic binary segmentation of our cells as in the original U-Net paper.
2. To track cells from one movie frame to the next, and to identify cell divisions and mother/daughter cells.

To do so we implemented three versions of the network. The first two perform sequential segmentation and tracking: One is a segmentation U-Net that is similar to the one introduced in the original paper (with pixel-wise weighted binary cross-entropy as loss function to enforce boundary detection between cells). The other performs tracking by taking both the segmentation results of the current frame and of the previous frame as an input.
The thrid version of the network performs segmentation and tracking at the same time by taking only the segmentation results of the previous frame as an input an returns the position of the cells in the new frame.

## Install
### GPU Requirements
While this code can run on CPU, we recommend running it on GPU for faster processing. To do so you will need [an nvidia gpu with a cuda compute capability of 3.5 or higher](https://developer.nvidia.com/cuda-gpus). Make sure you update your gpu drivers ([from the nvidia website](https://www.nvidia.com/Download/index.aspx?lang=en-us))

### Tensorflow 2.0 - GPU (recommended)
We recommend installing tensorflow 2.0 (GPU)  with [anaconda](https://www.anaconda.com/distribution/). See [here](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) for more details.

    (base)$ conda create --name tf_gpu python=3.6 tensorflow-gpu
The code should work with python >= 3.6, but just to be on the safe side we recommend you use the python=3.6 option when creating the tf_gpu environment. You can name your environment tf_gpu or whatever you like.

You will also need to install scikit-image in the environment:

    (base)$ conda activate tf_gpu
	(tf_gpu)$ conda install -c conda-forge scikit-image

And that's it! Although our code actually uses the Keras framework, since it is now included with tensorflow 2.0 as an API, you don't need to actually install it.

Once all of this is done you should be able to run training_seg.py from within the tf_gpu environment. You might run into memory allocation problems on your gpu, in which case you will need to change the batch size in the first lines of the script. We tested our code both on windows (10) and linux (ubuntu 18.04).

### Tensorflow 2.0 - CPU
If you can not install the GPU version of tensorflow (unfortunately the AMD graphics cards in Macs are not supported), you can install the CPU version and the code will run just fine. It will be significantly slower (probably by one to two orders of magnitude).

    (base)$ conda create --name tf_cpu python=3.6 tensorflow
    (base)$ conda activate tf_cpu
	(tf_cpu)$ conda install -c conda-forge scikit-image
	
Again you can name your environment whatever you like. 
	
### Elastic deformations
If you want to use random elastic deformations as described in Ronneberger et al (necessary for the yeast branch of this repository) you will need to install the relevant library from github:

    (base)$ conda activate tf_gpu
	(tf_gpu)$ conda install pip git
	(tf_gpu)$ pip install git+https://github.com/gvtulder/elasticdeform
	
Note: On Windows, you might need to [install c++ support for Visual Studio](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation)

## Example run:
### Training
You can train the segmentation and tracking networks by running the training scripts (after activating the tf_gpu environment and from within the repository):

    (tf_gpu)$ python train_seg.py
	(tf_gpu)$ python train_track.py
These two operations will probably take a while. About a couple hours each on standard consumer-grade GPUs.
The two scripts will save the models to hdf5 files `unet_moma_seg.hdf5` and `unet_moma_track.hdf5` at the root of the repository. They will also perform prediction on small test samples, the outputs of which can be found in `./data/segmentation/output/` and `./data/tracking/output/` respectively.

### Prediction
Once your two models are trained, you can run predictions on larger datasets using the `segmentation.py` and `tracking.py` scripts sequentially.

You can download a sample dataset [here]() and unzip it.

You will need to modify the two scripts to specify where to read and write images:
 * `segmentation.py`
   * `input_folder` must point to the dataset containing the images you downloaded
   * `save_folder` must point to a folder _that already exists_ for the script to save the segmentation outputs
 * `tracking.py`
   * `images_folder` must point to the dataset containing the images you downloaded (same as `input_folder` above)
   * `segmentation_folder` must point to the segmentation output folder (same as `save_folder` above)
   * `save_folder` must point to a folder _that already exists_ for the script to save the tracking outputs

Finally, you can run the two scripts one after the other:

    (tf_gpu)$ python segmentation.py
	(tf_gpu)$ python tracking.py

## Code structure
The repository contains code for handling data, training the networks, and for applying trained network models to prediction problems. It also contains training datasets in the `./data/` folder, but it doesn't contain sample datasets for prediction because it would take too much space. It also does not contain trained models. Both larger sample prediction sets and trained models are available [here]().

### Data processing (data.py)
The `data.py` file contains the input/output handling functions. It bundles together functions to read images from files, perform data augmentation on those images and concatenate them into the necessary formats for feeding into the different U-Net architectures.

#### Generators
To reduce the memory footprint and make on-the-fly data augmentation possible, we use generators to feed data into our models bit by bit. More info on generators [here](https://realpython.com/introduction-to-python-generators/)

There are two types of generators in the `data.py` file: Training ones and prediction ones:
##### Training generators
The training ones will read and output both the inputs of the U-Net and the expected outputs so that it can be fitted against them. The training ones will also perform data augmentation if operations are specified. (see next section)

**trainGenerator_seg()** and **trainGenerator_track()** generate training data for the segmentation and tracking versions respectively. They are structured basically the same way, but:
* **trainGenerator_seg** will yield a 2-elements tuple of: one M-by-N-by-1 numpy array containing the grayscale images of the cells and one M-by-N-by-2 numpy array of concatenated binary output mask + pixel-wise weights. [M-by-N being the desired input image shape, in our case 256-by-32]
The pixel-wise weights will be used in the weighted_cross_entropy() function of `model.py` (see below)
* **trainGenerator_track** will yield a 2-elements tuple of: one M-by-N-by-4 numpy array of concatenated the grayscale image of the cells in the current frame + the segmented cell of interest in the previous frame +  the grayscale image of the cells in the previous frame + the segmentation of all cells in the current frame, and one M-by-N-by-3 numpy array of concatenated binary output masks for the cell being tracked + the mask for the potential daughter + the background (everything else in the image, necessary for categorical cross-entropy).

##### Prediction generators
Prediction generators **predictGenerator_seg** and **predictGenerator_track** will only yield the inputs for the U-Net and are meant to be used once the model has been trained. They do not perform any data augmentation.

An additional prediction generator, **predictCompilefromseg_track**, for tracking creates the inputs as expected by the tracking U-Net from the output of the segmentation U-Net. To do so it will, for each segmentation output mask, identify the segmentation mask of the previous frame (as well as the current image frame and previous image frame) and for each cell in the previous frame create a sample for identifying where that cell is in the new segmentation output mask. It returns both the K-by-M-by-N-by-4 input tensor to feed into the tracking U-Net (with K being the number of samples) and a list of filenames to save the outputs of the tracking U-Net to.

For this to work, the files must use the same nomenclature as we do, i.e. a name of the form:
> Position\*\*\_Chamber\*\*\_Frame\*\*\*\.png

And the output will be saved as:
> Position\*\*\_Chamber\*\*\_Frame\*\*\*\_Cell\*\*.png

With the cell suffix corresponding to the cell position in the chamber, from top to bottom. (Hence mother cells will have files ending in _Cell01.png)

Please note that the size of the output, i.e. the number of input samples (K) that will be fed into U-Net, will vary for different segmented masks because the number of cells per chamber will vary.

Please also note that **predictCompilefromseg_track** is actually not a generator, it compiles a certain amount of data, returns it, and then exits.

#### Data augmentation
Data augmentation is the process of artificially increasing training set size by applying random image transformation operations. It usually reduces overfitting and allows for better generalization.

While Keras offers data augmentation routines through the ImageDataGenerator function, I wasn't able to get quite what i wanted out of it (surely because I am not familiar with it enough). I therefore wrote a couple operations that seemed relevant for our purposes using the scikit-image library. Those are called by the **data_augmentation()** function

The same operation is applied to all images provided as an input list. A dictionary containing the operations to perform and their range should also be provided as input. Finally, the order of the operations should also be provided: Typically 0 for binary images and 1 for grayscale. An order of 0 specifies that the value of the nearest pixel should be used and an order of 1 specifies that linear combination of nearby pixel values should be used.

### Model definitions (model.py)
The `model.py` file contains definitions of the U-Net models and of the custom loss for pixel-wise weighted binary cross entropy.

The **unet()** function defines a "generic" U-Net model. The dimensionality of the input, output and the activation function for the final layer are the only things that change between segmentation and tracking, and those can be passed as arguments.

**unet_seg()** calls the generic U-Net function with parameters for segmentation. It defines the loss as the **weighted_cross_entropy()**. This custom loss computes a binary cross-entropy, but attributes more weight to errors for pixels in between cells. This forces our network to learn to segment border pixels properly. See the original U-Net paper for more details. We used the mathematical formula described in the paper to generate those weighing map images. Finally, a custom metric, **unstack_acc()**, is also used. This additional metric is only used for reference and does not play an actual role in the fitting. It is basically the Keras binary accuracy that is being applied to the segmentation mask, with the weights map being discarded.

**unet_track()** calls the generic U-Net function with parameters for tracking. It defines the loss as categorical cross-entropy.

### Segmentation

![](https://gitlab.com/dunloplab/deeplearning/unet_moma/raw/images/unet_seg.png)

The segmentation U-Net is the same as the one presented in the original U-Net paper. It takes images as input and outputs a binary segmented image. The loss function had to be rewritten to be able to attribute different weights to different pixels in the output. (Note: This is actually exactly what is done in the original U-Net paper)

#### Training (train_seg.py)
The `training.py` script will call the **trainGenerator_seg** generator on the segmentation training set in `./data/segmentation/train/`. It will then get the U-Net model for segmentation from **unet_seg()** and fit it. Only the best fit is saved to disk to the `unet_moma_seg.hdf5` file. A test is run on the segmentation test set in `./data/segmentation/test/` and the result is saved to `./data/segmentation/output/`

#### Prediction (segmentation.py)
Once the training has been done, the model saved to disk `unet_moma_seg.hdf5` (or any other name you may have decided to use during training) can be used  to run prediction on new datasets with the `segmentation.py` script.
Please make sure you change the `input_folder` and `save_folder` variables prior to use.

Please note that the output can be used as the input for the tracking U-Net.

### Tracking

![](https://gitlab.com/dunloplab/deeplearning/unet_moma/raw/images/unet_track.png)

The tracking U-Net is similar to the segmentation one in its structure. However, it takes as inputs the grayscale image of the current frame and the previous frame, the segmentation mask of all cells in the current frame, and the segmentation of _a single cell in the previous frame_. The goal is to identify where this 'cell of interest' or 'seed' is in the current segmentation mask, and identifying any potential daughter.

As such it outputs two images of interest, the 'mother' and the 'daughter'. The latter will most of the time be entirely black and only feature a cell if a division has occured between the two frames. It actually outputs a third image which is the 'background' and is equal to the inverse of the mother and daughter images. This last image is necessary for computing the categorical cross entropy with Keras but is discarded when results are saved to disk.

#### Training (train_track.py)
Similarly to segmentation training, the `train_track.py` script calls the **trainGenerator_track** generator on the tracking training set in `./data/tracking/train/`. It will then get the U-Net model for tracking from **unet_track()** and fit it. Only the best fit is saved to disk to the `unet_moma_track.hdf5` file. A test is run on the tracking test set in `./data/tracking/test/` and the result is saved to `./data/tracking/output/`

#### Prediction (tracking.py)
Once the training has been done, the model saved to disk `unet_moma_track.hdf5` (or any other name you may have decided to use during training) can be used  to run tracking prediction on new datasets with the `tracking.py` script.
Please make sure you change the `images_folder`, `segmentation_folder` and `save_folder` variables prior to use.

Please note that this prediction script uses the **predictCompilefromseg_track** function. Your input files will therefore have to be named following our nomenclature:
> Position\*\*\_Chamber\*\*\_Frame\*\*\*\.png

### Simultaneous segmentation and tracking
#### Training
#### Prediction

## Troubleshooting
### OOM - Out of memory (GPU)
If you run the code on your GPU, you might run into memory problems. This is both linked to the batch size and the size of the images you are using. The batch size is fairly straightforward to change, just lower the value at the beginning of the `train_seg.py` or `train_track.py` files. Note that lower batch sizes may mean slower training convergence or lower performance overall. You might want to increase the number of epochs or steps per epoch to compensate.

The other solution would be to use a smaller image target size. However, if the original images are, say 512×512, downsizing it to 256×256 will certainly reduce the memory footprint, but it might cause some borders between cells in the binary masks to disappear. Instead, you would have to modify your images size upstream of U-Net to make sure that your training set does feature cell-to-cell borders in the segmentation masks.