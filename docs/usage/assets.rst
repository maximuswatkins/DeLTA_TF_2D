
Downloading assets
=================================

The assets module allows you to download the latest models, training sets, and
evaluation movies from `our google drive folder <https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP?usp=sharing>`_

You can download assets interactively simply by executing the following line
in python after importing delta::

    delta.assets.download_assets()

For first installation, we recommend downloading all assets and setting the
configuration level to 'global'. See :doc:`Configuration files <config>` 
for more information. But you can also specify which assets to download, 
or where to save them::

    delta.assets.download_assets(
        load_models=True,
        load_sets='/home/jeanbaptiste/data/delta_trainingsets',
        load_evals=False,
        config_level='local'
    )

Please note that some of these datasets are several GBs in size and the 
download process can take a while. By default they are saved in an 'assets' 
folder under the delta installation path.

`Source code <https://gitlab.com/dunloplab/delta/-/blob/master/delta/assets.py>`_

Signature:
----------

.. autofunction:: delta.assets.download_assets
