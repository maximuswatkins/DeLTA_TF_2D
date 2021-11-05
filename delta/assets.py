#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:34:55 2021

@author: jeanbaptiste
"""
import os
import requests

from . import __file__ as delta_init #Path to __init__ file
DELTA_DIR = os.path.dirname(delta_init)

LATEST_MODELS={
    'unet_moma_seg_multisets.hdf5':'1o4Ur0bQ7IvqCkfjeNFYdoEjpAsN7b2uK',
    'unet_moma_track_v2.hdf5':'1IRzCcP0mjdRRZ4Xzj9qSw4QI958apdTS',
    'unet_pads_seg.hdf5':'1PSqYtEdJJNaxv7xwDmcZ7X2JomnlF',
    'unet_pads_track.hdf5':'1EPRV-vF5Dv-UDOn0fdnpThIgjWxNZDio',
    'chambers_id_tessiechamp.hdf5':'13SFG0T73XG8NfVVwLyGRJ_smZdMiWLM_'
    }

def download_assets(load_models=None,config_file=None):
    
    # Should we download models?
    if load_models is None:
        load_models = _ask('Download latest models?')

    if load_models:
        model_files = _download_models(load_models)
    else:
        model_files = dict()
    
    # Should we write a new config file?
    if config_file is None:
        config_file = _ask('Write a new config.py file pointing to these new assets?')
    
    if config_file:
        _write_config(config_file,model_files=model_files)

def _ask(prompt,options={'y':True,'n':False}):

    answer=None
    prompt+=' ['
    for k in options:
        prompt+=k+'/'
    prompt=''.join(list(prompt)[:-1])+']: '

    while answer is None:

        answer = input(prompt)
        
        for k,v in options.items():
            if answer==k:
                return v

        answer=None
        print("invalid answer: '%s'"%answer)

def _write_config(config_file, model_files=dict()):
    
    

def _download_models(models_dir=None):
    
    if type(models_dir)!=str:
        # Retrieve the path of the delta install folder:
        models_dir = os.path.join(DELTA_DIR,'assets','models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    print('Models will be downloaded to %s'%models_dir)
    
    model_files=dict()
    for filename, google_id in LATEST_MODELS.items():
        print('Downloading model: %s...'%filename)
        model_files[filename] = os.path.join(models_dir,filename)
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
