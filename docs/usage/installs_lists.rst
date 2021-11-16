Successful installs
====================

Here we maintain a list of successful installation
library lists for debugging purposes.

These environments installed successfully, detected
the machine's GPU, and passed
all our tests with pytest.

If you had to tweak library versions, or even if you
had a smooth installation process, feel free to 
share your environment list with us via gitlab's issues 
so we can list it here.

.. highlight:: none

Windows 10 pip install - TF 2.7
-------------------------------

install: 2021-11-14

First installed CUDA 11.5 and cuDNN 8.3

Created dedicated conda env with python=3.9

`pip list` output::

	Package                      Version
	---------------------------- ---------
	absl-py                      1.0.0
	astunparse                   1.6.3
	cachetools                   4.2.4
	certifi                      2021.10.8
	charset-normalizer           2.0.7
	cycler                       0.11.0
	ffmpeg-python                0.2.0
	flatbuffers                  2.0
	future                       0.18.2
	gast                         0.4.0
	google-auth                  2.3.3
	google-auth-oauthlib         0.4.6
	google-pasta                 0.2.0
	grpcio                       1.41.1
	h5py                         3.5.0
	idna                         3.3
	imageio                      2.10.4
	keras                        2.7.0
	Keras-Preprocessing          1.1.2
	kiwisolver                   1.3.2
	libclang                     12.0.0
	Markdown                     3.3.4
	matplotlib                   3.4.3
	networkx                     2.6.3
	numpy                        1.21.4
	oauthlib                     3.1.1
	opencv-python                4.5.4.58
	opt-einsum                   3.3.0
	Pillow                       8.4.0
	pip                          21.2.4
	protobuf                     3.19.1
	pyasn1                       0.4.8
	pyasn1-modules               0.2.8
	pyparsing                    3.0.6
	python-dateutil              2.8.2
	PyWavelets                   1.2.0
	requests                     2.26.0
	requests-oauthlib            1.3.0
	rsa                          4.7.2
	scikit-image                 0.18.3
	scipy                        1.7.2
	setuptools                   58.0.4
	six                          1.16.0
	tensorboard                  2.7.0
	tensorboard-data-server      0.6.1
	tensorboard-plugin-wit       1.8.0
	tensorflow                   2.7.0
	tensorflow-estimator         2.7.0
	tensorflow-io-gcs-filesystem 0.22.0
	termcolor                    1.1.0
	tifffile                     2021.11.2
	typing_extensions            4.0.0
	urllib3                      1.26.7
	Werkzeug                     2.0.2
	wheel                        0.37.0
	wincertstore                 0.2
	wrapt                        1.13.3

Windows 10 conda install, TF 2.3
---------------------------------

install date: End of 2020	

`conda list` output::
	# packages in environment at C:\Users\jeanbaptiste\Anaconda3\envs\delta_env:
	#
	# Name                    Version                   Build  Channel
	_tflow_select             2.3.0                       gpu
	absl-py                   0.12.0             pyhd8ed1ab_0    conda-forge
	aiohttp                   3.7.4            py38h294d835_0    conda-forge
	alabaster                 0.7.12                     py_0    conda-forge
	appdirs                   1.4.4              pyh9f0ad1d_0    conda-forge
	argh                      0.26.2          pyh9f0ad1d_1002    conda-forge
	arrow                     0.13.1                   py38_0
	astor                     0.8.1              pyh9f0ad1d_0    conda-forge
	astroid                   2.5.1            py38haa244fe_0    conda-forge
	astunparse                1.6.3              pyhd8ed1ab_0    conda-forge
	async-timeout             3.0.1                   py_1000    conda-forge
	async_generator           1.10                       py_0    conda-forge
	atomicwrites              1.4.0              pyh9f0ad1d_0    conda-forge
	attrs                     20.3.0             pyhd3deb0d_0    conda-forge
	autopep8                  1.5.6              pyhd8ed1ab_0    conda-forge
	babel                     2.9.0              pyhd3deb0d_0    conda-forge
	backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
	backports                 1.0                        py_2    conda-forge
	backports.functools_lru_cache 1.6.1                      py_0    conda-forge
	bcrypt                    3.2.0            py38h294d835_1    conda-forge
	binaryornot               0.4.4              pyhd3eb1b0_1
	black                     20.8b1                     py_1    conda-forge
	bleach                    3.3.0              pyh44b312d_0    conda-forge
	blinker                   1.4                        py_1    conda-forge
	blosc                     1.21.0               h0e60522_0    conda-forge
	boto3                     1.17.35                  pypi_0    pypi
	botocore                  1.20.35                  pypi_0    pypi
	brotli                    1.0.9                h0e60522_4    conda-forge
	brotlipy                  0.7.0           py38h294d835_1001    conda-forge
	bzip2                     1.0.8                h8ffe710_4    conda-forge
	ca-certificates           2021.7.5             haa95532_1
	cachetools                4.2.1              pyhd8ed1ab_0    conda-forge
	certifi                   2021.5.30        py38haa95532_0
	cffi                      1.14.5           py38hd8c33c5_0    conda-forge
	chardet                   4.0.0            py38haa244fe_1    conda-forge
	charls                    2.2.0                h39d44d4_0    conda-forge
	click                     7.1.2              pyh9f0ad1d_0    conda-forge
	cloudpickle               1.6.0                      py_0    conda-forge
	colorama                  0.4.4              pyh9f0ad1d_0    conda-forge
	cookiecutter              1.7.2              pyhd3eb1b0_0
	cryptography              3.4.6            py38hd7da0ea_0    conda-forge
	cudatoolkit               10.1.243             h3826478_8    conda-forge
	cudnn                     7.6.5.32             h36d860d_1    conda-forge
	cycler                    0.10.0                     py_2    conda-forge
	cytoolz                   0.11.0           py38h294d835_3    conda-forge
	dask-core                 2021.3.0           pyhd8ed1ab_0    conda-forge
	dataclasses               0.8                pyhc8e2a94_1    conda-forge
	decorator                 4.4.2                      py_0    conda-forge
	defusedxml                0.7.1              pyhd8ed1ab_0    conda-forge
	diff-match-patch          20200713           pyh9f0ad1d_0    conda-forge
	docutils                  0.16             py38haa244fe_3    conda-forge
	elasticdeform             0.4.9                    pypi_0    pypi
	entrypoints               0.3             pyhd8ed1ab_1003    conda-forge
	ffmpeg                    4.2.2                he774522_0
	ffmpeg-python             0.2.0                      py_0    conda-forge
	flake8                    3.8.4                      py_0    conda-forge
	freeglut                  3.2.1                h0e60522_2    conda-forge
	freetype                  2.10.4               h546665d_1    conda-forge
	future                    0.18.2           py38haa244fe_3    conda-forge
	gast                      0.4.0              pyh9f0ad1d_0    conda-forge
	giflib                    5.2.1                h8d14728_2    conda-forge
	git                       2.30.2               h57928b3_0    conda-forge
	google-auth               1.24.0             pyhd3deb0d_0    conda-forge
	google-auth-oauthlib      0.4.1                      py_2    conda-forge
	google-pasta              0.2.0              pyh8c360ce_0    conda-forge
	grpcio                    1.36.1           py38he5377a8_0    conda-forge
	h5py                      2.10.0          nompi_py38h6053941_105    conda-forge
	hdf5                      1.10.6          nompi_h5268f04_1114    conda-forge
	helpdev                   0.7.1              pyhd8ed1ab_0    conda-forge
	icu                       68.1                 h0e60522_0    conda-forge
	idna                      2.10               pyh9f0ad1d_0    conda-forge
	imagecodecs               2021.1.28        py38hb9201fa_0    conda-forge
	imageio                   2.9.0                      py_0    conda-forge
	imagesize                 1.2.0                      py_0    conda-forge
	importlib-metadata        3.7.3            py38haa244fe_0    conda-forge
	importlib_metadata        3.7.3                hd8ed1ab_0    conda-forge
	inflection                0.5.1            py38haa95532_0
	intel-openmp              2020.3             h57928b3_311    conda-forge
	intervaltree              3.0.2                      py_0    conda-forge
	ipykernel                 5.5.0            py38hc5df569_1    conda-forge
	ipython                   7.21.0           py38hc5df569_0    conda-forge
	ipython_genutils          0.2.0                      py_1    conda-forge
	isort                     5.8.0              pyhd8ed1ab_0    conda-forge
	jasper                    2.0.14               h77af90b_2    conda-forge
	jedi                      0.17.2           py38haa244fe_1    conda-forge
	jinja2                    2.11.3             pyh44b312d_0    conda-forge
	jinja2-time               0.2.0              pyhd3eb1b0_2
	jmespath                  0.10.0                   pypi_0    pypi
	jpeg                      9d                   h8ffe710_0    conda-forge
	jsonschema                3.2.0              pyhd8ed1ab_3    conda-forge
	jupyter_client            6.1.12             pyhd8ed1ab_0    conda-forge
	jupyter_core              4.7.1            py38haa244fe_0    conda-forge
	jupyterlab_pygments       0.1.2              pyh9f0ad1d_0    conda-forge
	jxrlib                    1.1                  h8ffe710_2    conda-forge
	keras-applications        1.0.8                      py_1    conda-forge
	keras-preprocessing       1.1.2              pyhd8ed1ab_0    conda-forge
	keyring                   23.0.0           py38haa244fe_0    conda-forge
	kiwisolver                1.3.1            py38hbd9d945_1    conda-forge
	krb5                      1.17.2               hbae68bd_0    conda-forge
	lazy-object-proxy         1.6.0            py38h294d835_0    conda-forge
	lcms2                     2.12                 h2a16943_0    conda-forge
	lerc                      2.2.1                h0e60522_0    conda-forge
	libaec                    1.0.4                h39d44d4_1    conda-forge
	libblas                   3.9.0                     8_mkl    conda-forge
	libcblas                  3.9.0                     8_mkl    conda-forge
	libclang                  11.1.0          default_h5c34c98_0    conda-forge
	libcurl                   7.75.0               hf1763fc_0    conda-forge
	libdeflate                1.7                  h8ffe710_5    conda-forge
	liblapack                 3.9.0                     8_mkl    conda-forge
	liblapacke                3.9.0                     8_mkl    conda-forge
	libopencv                 4.5.1            py38hf7032e7_0    conda-forge
	libpng                    1.6.37               h1d00b33_2    conda-forge
	libprotobuf               3.15.6               h7755175_0    conda-forge
	libsodium                 1.0.18               h8d14728_1    conda-forge
	libspatialindex           1.9.3                h39d44d4_3    conda-forge
	libssh2                   1.9.0                h680486a_6    conda-forge
	libtiff                   4.2.0                hc10be44_0    conda-forge
	libwebp-base              1.2.0                h8ffe710_2    conda-forge
	libzopfli                 1.0.3                h0e60522_0    conda-forge
	lz4-c                     1.9.3                h8ffe710_0    conda-forge
	m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
	m2w64-gcc-libs            5.3.0                         7    conda-forge
	m2w64-gcc-libs-core       5.3.0                         7    conda-forge
	m2w64-gmp                 6.1.0                         2    conda-forge
	m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
	markdown                  3.3.4              pyhd8ed1ab_0    conda-forge
	markupsafe                1.1.1            py38h294d835_3    conda-forge
	matplotlib-base           3.3.4            py38h34ddff4_0    conda-forge
	mccabe                    0.6.1                      py_1    conda-forge
	mistune                   0.8.4           py38h294d835_1003    conda-forge
	mkl                       2020.4             hb70f87d_311    conda-forge
	msys2-conda-epoch         20160418                      1    conda-forge
	multidict                 5.1.0            py38h294d835_1    conda-forge
	mypy_extensions           0.4.3            py38haa244fe_3    conda-forge
	nbclient                  0.5.3              pyhd8ed1ab_0    conda-forge
	nbconvert                 6.0.7            py38haa244fe_3    conda-forge
	nbformat                  5.1.2              pyhd8ed1ab_1    conda-forge
	nest-asyncio              1.4.3              pyhd8ed1ab_0    conda-forge
	networkx                  2.5                        py_0    conda-forge
	numpy                     1.20.1           py38h0cc643e_0    conda-forge
	numpydoc                  1.1.0                      py_1    conda-forge
	oauthlib                  3.0.1                      py_0    conda-forge
	olefile                   0.46               pyh9f0ad1d_1    conda-forge
	opencv                    4.5.1            py38haa244fe_0    conda-forge
	openjpeg                  2.4.0                h48faf41_0    conda-forge
	openssl                   1.1.1l               h2bbff1b_0
	opt_einsum                3.3.0                      py_0    conda-forge
	packaging                 20.9               pyh44b312d_0    conda-forge
	pandoc                    2.13                 h8ffe710_0    conda-forge
	pandocfilters             1.4.2                      py_1    conda-forge
	paramiko                  2.7.2              pyh9f0ad1d_0    conda-forge
	parso                     0.7.0              pyh9f0ad1d_0    conda-forge
	pathspec                  0.8.1              pyhd3deb0d_0    conda-forge
	pexpect                   4.8.0              pyh9f0ad1d_2    conda-forge
	pickleshare               0.7.5                   py_1003    conda-forge
	pillow                    8.1.2            py38h9273828_0    conda-forge
	pip                       21.0.1             pyhd8ed1ab_0    conda-forge
	pluggy                    0.13.1           py38haa244fe_4    conda-forge
	pooch                     1.3.0              pyhd8ed1ab_0    conda-forge
	poyo                      0.5.0              pyhd3eb1b0_0
	prompt-toolkit            3.0.18             pyha770c72_0    conda-forge
	protobuf                  3.15.6           py38h885f38d_0    conda-forge
	psutil                    5.8.0            py38h294d835_1    conda-forge
	ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
	py-opencv                 4.5.1            py38hc5df569_0    conda-forge
	pyasn1                    0.4.8                      py_0    conda-forge
	pyasn1-modules            0.2.7                      py_0    conda-forge
	pycodestyle               2.6.0              pyh9f0ad1d_0    conda-forge
	pycparser                 2.20               pyh9f0ad1d_2    conda-forge
	pydocstyle                6.0.0              pyhd8ed1ab_0    conda-forge
	pyflakes                  2.2.0              pyh9f0ad1d_0    conda-forge
	pygments                  2.8.1              pyhd8ed1ab_0    conda-forge
	pyjwt                     2.0.1              pyhd8ed1ab_0    conda-forge
	pylint                    2.7.2            py38haa244fe_0    conda-forge
	pyls-black                0.4.6              pyh9f0ad1d_0    conda-forge
	pyls-spyder               0.3.2              pyhd8ed1ab_0    conda-forge
	pynacl                    1.4.0            py38h31c79cd_2    conda-forge
	pyopenssl                 20.0.1             pyhd8ed1ab_0    conda-forge
	pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
	pyqt                      5.12.3           py38haa244fe_7    conda-forge
	pyqt-impl                 5.12.3           py38h885f38d_7    conda-forge
	pyqt5-sip                 4.19.18          py38h885f38d_7    conda-forge
	pyqtchart                 5.12             py38h885f38d_7    conda-forge
	pyqtwebengine             5.12.1           py38h885f38d_7    conda-forge
	pyreadline                2.1             py38haa244fe_1003    conda-forge
	pyrsistent                0.17.3           py38h294d835_2    conda-forge
	pysocks                   1.7.1            py38haa244fe_3    conda-forge
	python                    3.8.8           h7840368_0_cpython    conda-forge
	python-bioformats         4.0.4                    pypi_0    pypi
	python-dateutil           2.8.1                      py_0    conda-forge
	python-javabridge         4.0.3                    pypi_0    pypi
	python-jsonrpc-server     0.4.0              pyh9f0ad1d_0    conda-forge
	python-language-server    0.36.2             pyhd8ed1ab_0    conda-forge
	python-slugify            5.0.2              pyhd3eb1b0_0
	python_abi                3.8                      1_cp38    conda-forge
	pytz                      2021.1             pyhd8ed1ab_0    conda-forge
	pywavelets                1.1.1            py38h347fdf6_3    conda-forge
	pywin32                   300              py38h294d835_0    conda-forge
	pywin32-ctypes            0.2.0           py38haa244fe_1003    conda-forge
	pyyaml                    5.4.1            py38h294d835_0    conda-forge
	pyzmq                     22.0.3           py38h09162b1_1    conda-forge
	qdarkstyle                3.0.2              pyhd3eb1b0_0
	qstylizer                 0.1.10             pyhd3eb1b0_0
	qt                        5.12.9               h5909a2a_4    conda-forge
	qtawesome                 1.0.2              pyhd8ed1ab_0    conda-forge
	qtconsole                 5.0.3              pyhd8ed1ab_0    conda-forge
	qtpy                      1.9.0                      py_0    conda-forge
	regex                     2021.3.17        py38h294d835_0    conda-forge
	requests                  2.25.1             pyhd3deb0d_0    conda-forge
	requests-oauthlib         1.3.0              pyh9f0ad1d_0    conda-forge
	rope                      0.18.0             pyh9f0ad1d_0    conda-forge
	rsa                       4.7.2              pyh44b312d_0    conda-forge
	rtree                     0.9.7            py38h8b54edf_1    conda-forge
	s3transfer                0.3.6                    pypi_0    pypi
	scikit-image              0.18.1           py38h4c96930_0    conda-forge
	scipy                     1.6.1            py38h5f893b4_0    conda-forge
	setuptools                49.6.0           py38haa244fe_3    conda-forge
	six                       1.15.0             pyh9f0ad1d_0    conda-forge
	snappy                    1.1.8                ha925a31_3    conda-forge
	snowballstemmer           2.1.0              pyhd8ed1ab_0    conda-forge
	sortedcontainers          2.3.0              pyhd8ed1ab_0    conda-forge
	sphinx                    3.5.3              pyhd8ed1ab_0    conda-forge
	sphinxcontrib-applehelp   1.0.2                      py_0    conda-forge
	sphinxcontrib-devhelp     1.0.2                      py_0    conda-forge
	sphinxcontrib-htmlhelp    1.0.3                      py_0    conda-forge
	sphinxcontrib-jsmath      1.0.1                      py_0    conda-forge
	sphinxcontrib-qthelp      1.0.3                      py_0    conda-forge
	sphinxcontrib-serializinghtml 1.1.4                      py_0    conda-forge
	spyder                    5.0.0            py38haa95532_1
	spyder-kernels            2.0.5            py38haa95532_0
	sqlite                    3.35.2               h8ffe710_0    conda-forge
	tensorboard               2.4.1              pyhd8ed1ab_0    conda-forge
	tensorboard-plugin-wit    1.8.0              pyh44b312d_0    conda-forge
	tensorflow                2.3.0           mkl_py38h1fcfbd6_0
	tensorflow-base           2.3.0           gpu_py38h7339f5a_0
	tensorflow-estimator      2.4.0              pyh9656e83_0    conda-forge
	tensorflow-gpu            2.3.0                he13fc11_0
	termcolor                 1.1.0                      py_2    conda-forge
	testpath                  0.4.4                      py_0    conda-forge
	text-unidecode            1.3                pyhd3eb1b0_0
	textdistance              4.2.1              pyhd8ed1ab_0    conda-forge
	three-merge               0.1.1              pyh9f0ad1d_0    conda-forge
	tifffile                  2021.3.17          pyhd8ed1ab_0    conda-forge
	tinycss                   0.4             pyhd3eb1b0_1002
	tk                        8.6.10               h8ffe710_1    conda-forge
	toml                      0.10.2             pyhd8ed1ab_0    conda-forge
	toolz                     0.11.1                     py_0    conda-forge
	tornado                   6.1              py38h294d835_1    conda-forge
	traitlets                 5.0.5                      py_0    conda-forge
	typed-ast                 1.4.2            py38h294d835_0    conda-forge
	typing-extensions         3.7.4.3                       0    conda-forge
	typing_extensions         3.7.4.3                    py_0    conda-forge
	ujson                     4.0.2            py38h885f38d_0    conda-forge
	unidecode                 1.2.0              pyhd3eb1b0_0
	urllib3                   1.26.4             pyhd8ed1ab_0    conda-forge
	vc                        14.2                 hb210afc_4    conda-forge
	vs2015_runtime            14.28.29325          h5e1d092_4    conda-forge
	watchdog                  1.0.2            py38haa244fe_1    conda-forge
	wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
	webencodings              0.5.1                      py_1    conda-forge
	werkzeug                  1.0.1              pyh9f0ad1d_0    conda-forge
	wheel                     0.36.2             pyhd3deb0d_0    conda-forge
	whichcraft                0.6.1              pyhd3eb1b0_0
	win_inet_pton             1.1.0            py38haa244fe_2    conda-forge
	wincertstore              0.2             py38haa244fe_1006    conda-forge
	wrapt                     1.12.1           py38h294d835_3    conda-forge
	xz                        5.2.5                h62dcd97_1    conda-forge
	yaml                      0.2.5                he774522_0    conda-forge
	yapf                      0.30.0             pyh9f0ad1d_0    conda-forge
	yarl                      1.6.3            py38h294d835_1    conda-forge
	zeromq                    4.3.4                h0e60522_0    conda-forge
	zfp                       0.5.5                h0e60522_4    conda-forge
	zipp                      3.4.1              pyhd8ed1ab_0    conda-forge
	zlib                      1.2.11            h62dcd97_1010    conda-forge
	zstd                      1.4.9                h6255e5f_0    conda-forge