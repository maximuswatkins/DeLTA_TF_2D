
Assets
=================================

`Source code <https://gitlab.com/dunloplab/delta/-/blob/master/delta/assets.py>`_ |
:doc:`API <../_generated/delta.assets>`

We provide archives of our trained models, training sets, and evaluation movies
in `this google drive folder <https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP?usp=sharing>`_.
We refer to these data files as "assets".

You can download them manually, but the assets are now downloaded automatically
the first time you try to use them via the
`pooch library <https://www.fatiando.org/pooch/latest/>`_. This means you do
not need to call the `download_datasets()` function anymore.

If the assets changed, either because you overwrote them or because we updated
them online, they will be re-downloaded automatically. If you wish to prevent
them from being downloaded automatically, you can specify their paths in the
config (see below).


Cache location
--------------------

By default, assets will be downloaded under the cache locations used by the
`pooch library <https://www.fatiando.org/pooch/latest/api/generated/pooch.os_cache.html>`_,
for example under `C:/Users/jeanbaptiste/AppData/Local/delta/delta/Cache`
on Windows. If you wish for delta to use a different cache location you
will need to set the `DELTA_ASSETS_CACHE` environment variable to point to
where you want this data to be downloaded.

For example you can
`set this permanently within your conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables>`_::

    conda env config vars set DELTA_ASSETS_CACHE="D:/data/delta_cache"



Local files, automated downloads, and `config`
--------------------------------------------------

By default, the `config` variables for the assets files and folders point to
`None`. This means the `assets` module and the `pooch` library handle assets
management. However, if you want to use your own custom trained models or
training sets, or if you want to disable automated model downloads, you can
specify their paths in the :doc:`config module <config_desc>`. For example::

    delta.config.model_file_rois = 'D:/data/delta_cache/unet_momachambers_seg.hdf5'

Or, if you want this change to be more permanent you can modify the relevant
config file directly.
