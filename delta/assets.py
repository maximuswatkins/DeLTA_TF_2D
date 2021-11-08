#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:34:55 2021

@author: jeanbaptiste
"""
import os
import requests
import json

from .config import _DELTA_DIR, _DEFAULTS_2D, _DEFAULTS_mothermachine

LATEST_MODELS={
    'unet_moma_seg':'1o4Ur0bQ7IvqCkfjeNFYdoEjpAsN7b2uK',
    'unet_moma_track':'1IRzCcP0mjdRRZ4Xzj9qSw4QI958apdTS',
    'unet_pads_seg':'1PSqYtEdJJNaxv7xwDmcZ7X2JomnlF-dP',
    'unet_pads_track':'1EPRV-vF5Dv-UDOn0fdnpThIgjWxNZDio',
    'unet_momachambers_seg':'13SFG0T73XG8NfVVwLyGRJ_smZdMiWLM_'
    }

def download_assets(load_models=None,config_level=None):
    '''
    Download assets such as latest models, training sets etc

    Parameters
    ----------
    load_models : bool, str, or None, optional
        If True, the models will be downloaded from 
        https://drive.google.com/drive/u/0/folders/14EvBCMJIk2LpBCQL-jvsuEwUHNYamyyN
        into an assets folder in the library install path. If a path str is provided,
        the models will be downloaded to the folder it points to. If None, an
        interactive prompt will be used.
        The default is None.
    config_level : bool, str, or None, optional
        If False, no config file is written. If True, same as 'local'
        If 'local', new config files will be written under the user's local \
        config folder, ~/.delta. If 'global', the config files will be written
        to the install folder, to be shared among all users. If a specific dir
        path is provided, the config files will be written there.
        If None, and interactive prompt will let the user pick.
        The default is None.

    Returns
    -------
    None.

    '''
    
    # Should we download models?
    if load_models is None:
        load_models = _ask('Download latest models?')

    if load_models:
        model_files = _download_models(load_models)
    else:
        model_files = dict()
    
    # Should we write a new config file?
    if config_level is None:
        config_level = _ask(
            'Write new config_.json files pointing to these new assets?',
            options={
                'local':'local',
                'global':'global',
                'no':False
                }
            )

    if config_level:
        _write_default_configs(model_files=model_files, config_level=config_level)

def _ask(prompt,options={'y':True,'n':False}):

    answer=None
    prompt+=' ['
    for k in options:
        prompt+=k+'/'
    prompt=''.join(list(prompt)[:-1])+']: '

    while answer is None:

        answer = input(prompt)
        
        for k,v in options.items():
            if answer.lower()==k:
                return v

        answer=None
        print("invalid answer: '%s'"%answer)

def _write_default_configs(config_level='local',model_files=dict()):
    
    if config_level=='local':
        config_dir = os.path.expanduser('~/.delta')
    elif config_level=='global':
        config_dir = os.path.join(_DELTA_DIR,'assets','config')
    elif os.path.exists(config_level):
        config_dir = config_level
    else:
        raise ValueError(
            '''Invalid configs level/path: %s
            If providing a folder path, the folder must exist
            '''%config_level
            )

    print('Config files will be written to %s'%config_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Write 2D config:
    config = _DEFAULTS_2D.copy()
    if len(model_files)>0:
        config['model_file_seg']=model_files['unet_pads_seg']
        config['model_file_track']=model_files['unet_pads_track']
    with open(os.path.join(config_dir,'config_2D.json'),'w') as f:
        json.dump(config, f, indent=4)
    
    # Write mothermachine config:
    config = _DEFAULTS_mothermachine.copy()
    if len(model_files)>0:
        config['model_file_rois']=model_files['unet_momachambers_seg']
        config['model_file_seg']=model_files['unet_moma_seg']
        config['model_file_track']=model_files['unet_moma_track']
    with open(os.path.join(config_dir,'config_mothermachine.json'),'w') as f:
        json.dump(config, f, indent=4)

def _download_models(model_ids=LATEST_MODELS, models_dir=None):
    '''
    Download models from gdrive

    Parameters
    ----------
    model_ids : dict, optional
        Dictionary of the filename:google_id for all necessary model files. 
        The default is LATEST_MODELS.
    models_dir : None or str, optional
        path to save folder. The default is None.

    Returns
    -------
    model_files : dict
        dictionary of save paths for each model.

    '''
    
    if type(models_dir)!=str:
        # Retrieve the path of the delta install folder:
        models_dir = os.path.join(_DELTA_DIR,'assets','models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    print('Models will be downloaded to %s'%models_dir)
    
    model_files=dict()
    for filename, google_id in LATEST_MODELS.items():
        print('Downloading model: %s...'%filename)
        model_files[filename] = os.path.join(models_dir,filename+'.hdf5')
        _download_file_from_google_drive(google_id, model_files[filename])
    
    print('Done')
    return model_files

def _download_file_from_google_drive(id, destination):
    #https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = _get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    _save_response_content(response, destination)    

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
