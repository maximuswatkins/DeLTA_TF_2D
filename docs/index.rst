.. DeLTA documentation master file, created by
   sphinx-quickstart on Tue Nov  9 17:32:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeLTA's documentation!
=================================

DeLTA (Deep Learning for Time-lapse Analysis) is a deep learning-based image 
processing pipeline for segmenting and tracking single cells in time-lapse 
microscopy movies.

.. image:: https://gitlab.com/dunloplab/delta/-/raw/images/DeLTAexample.gif
    :alt: An illustration of DeLTA's performance on an agar pad movie of E. Coli
    :align: center

Version 2 preprint: 
`O’Connor, O. M., Alnahhas, R. N., Lugagne, J.-B., & Dunlop, M. J. (2021). DeLTA 2.0: A deep learning pipeline for quantifying single-cell spatial and temporal dynamics. BioRxiv, 2021.08.10.455795 <https://www.biorxiv.org/content/10.1101/2021.08.10.455795v1>`_

Version 1 paper:
`Lugagne, J.-B., Lin, H., & Dunlop, M. J. (2020). DeLTA: Automated cell segmentation, tracking, and lineage reconstruction using deep learning. PLOS Computational Biology, 16(4), e1007673. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673>`_

Gitlab repository: 
`https://gitlab.com/dunloplab/delta <https://gitlab.com/dunloplab/delta>`_

🐛 If you encounter bugs, have questions about the software, suggestions for new
features, or even comments
about the documentation, please use Gitlab's issue system

Overview
--------
DeLTA revolves around a pipeline that can process movies of rod-shaped bacteria 
growing in 2D setups 
such as agarose pads, as well as movies of E. coli cells trapped in a 
microfluidic device known as a "mother machine".
Our pipeline is centered around two U-Net neural networks that are used sequentially:

* To perform semantic binary segmentation of our cells as in the original U-Net paper.
* To track cells from one movie frame to the next, and to identify cell divisions and mother/daughter cells.

A third U-Net can be used to identify regions of interest in the image before perfoming segmentation. This is used with mother machine movies to identify single chambers, but by default only 1 ROI covering the entire field-of-view is used in the 2D version.
The U-Nets are implemented in Tensorflow 2 via the Keras API.

Getting started
----------------

You can quickly check how DeLTA performs on your data for free
by using our `Google Colab notebook <https://colab.research.google.com/drive/1UL9oXmcJFRBAm0BMQy_DMKg4VHYGgtxZ>`_


To get started on your own system, check out the 
:doc:`installation instructions <usage/installation>`,
download our :doc:`latest models <usage/assets_desc>`, and run the 
:doc:`pipeline <usage/pipeline_desc>` on our data or your own.

Contents
----------

.. toctree::
    :maxdepth: 2
    
    usage/installation
    usage/assets_desc
    usage/config_desc
    usage/pipeline_desc
    usage/outputs
    usage/data_desc
    usage/utils_desc
    usage/model_desc
    usage/scripts

API
----


Navigate our API below:

.. autosummary::
    :toctree: _generated
    :template: my_custom_module.rst
    :recursive:

    delta.assets

Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`