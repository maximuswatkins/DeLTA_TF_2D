#%% Setup
### Modules:
import sys, time, cv2, re, os
import numpy as np
from data import postprocess
from model import unet_chambers, unet_seg, unet_track
from utilities import *
from scipy.io import savemat

# DeLTA_data = '/home/jeanbaptiste/data/DeLTA_data/'
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'

### Models:
model_file_chambers = DeLTA_data + 'chambers_id/models/chambers_id_tessiechamp.hdf5'
model_file_seg = DeLTA_data + 'mother_machine/models/unet_moma_seg_paulsson.hdf5'
model_file_track = DeLTA_data + 'mother_machine/models/unet_moma_track_multisets.hdf5'
target_size_chambers = (512, 512)
input_size_chambers = target_size_chambers + (1,)
target_size_seg = (256, 32)

model_chambers = unet_chambers(input_size = input_size_chambers)
model_chambers.load_weights(model_file_chambers)

model_seg = unet_seg(input_size = target_size_seg + (1,))
model_seg.load_weights(model_file_seg)

model_track = unet_track(input_size = target_size_seg + (4,))
model_track.load_weights(model_file_track)

### XP File:
xpfile = DeLTA_data + 'mother_machine/evaluation/evaluation_movie.ome.tiff'
channelnames = ['trans','gfp'] # trans will be used for DeLTA, all other channels will be treated as fluo
xp = xpreader(xpfile,
              channelnames=channelnames,
              use_bioformats=False)

### Results folder:
resfolder = DeLTA_data  + 'mother_machine/evaluation/results/'
os.mkdir(resfolder)
os.mkdir(resfolder+'preprocessed')
os.mkdir(resfolder+'postprocessed')

# Misc:
rotation = 0 # For now we don't correct

#%% Run over all positions:

for pos in range(xp.positions):
    
    
    ##### Chambers identificaton:
    start_time = time.time()
    print('Pos. '+str(pos+1)+'/'+str(xp.positions)+': Chambers ID & Drift...', end='')
    transframes = xp.getframes(positions=pos,channels='trans',rescale=(0,1))
    firstframe = np.expand_dims(np.expand_dims(cv2.resize(transframes[0],(512,512)),axis=0),axis=3) # using expand_dims to get it into a shape that the chambers id unet accepts
    chambermask = model_chambers.predict(firstframe,verbose=0)
    chambermask = cv2.resize(np.squeeze(chambermask),transframes.shape[3:0:-1]) # scaling back to original size
    chamberboxes = getChamberBoxes(np.squeeze(chambermask))
    drifttemplate = getDriftTemplate(chamberboxes, transframes[0])
    driftcorbox = dict(xtl = 0,
                        xbr = None,
                        ytl = 0,
                        ybr = max(chamberboxes,key=lambda elem: elem['ytl'])['ytl']
                        )
    transframes, driftvalues = driftcorr(transframes, template=drifttemplate, box=driftcorbox)
    
    # Save pre-processing to disk:
    savemat(resfolder+'preprocessed/Position'+str(pos).zfill(6)+'.mat',
            {'tiffile': xpfile,
             'moviedimensions': [xp.y,xp.x,xp.channels,xp.timepoints],
             'proc': {'rotation': rotation,
                      'chambers': np.array([[c['xtl'],c['ytl'],c['xbr']-c['xtl'],c['ybr']-c['ytl']] for c in chamberboxes],dtype=np.float64),
                      'XYdrift': np.flip(np.transpose(np.array(-driftvalues)),axis=0)}})
    
    
    # Load up fluoresence images, apply drift correction:
    fluoframes = xp.getframes(positions=pos, channels=[x for x in channelnames if x!='trans'], squeeze_dimensions=False)[0] # Not squeezing in case there is only one fluo channel, but squeezing the first position dimension with [0]
    for f in range(fluoframes.shape[1]): #Go over all channels:
        fluoframes[:,f,:,:], _ = driftcorr(fluoframes[:,f,:,:], drift=driftvalues) # Apply drift correction
    
    ##### Cell segmentation:
    print('\b\b\b - Segmentation...', end='')
    seg_inputs = []
    # Compile segmentation inputs:
    for m, chamberbox in enumerate(chamberboxes):
        for i in range(transframes.shape[0]):
            seg_inputs.append(cv2.resize(rangescale(cropbox(transframes[i],chamberbox),(0,1)),(32,256)))
    seg_inputs = np.expand_dims(np.array(seg_inputs),axis=3) # Format into 4D tensor
    # Run segmentation U-Net:
    seg = model_seg.predict(seg_inputs,verbose=0)
    seg = postprocess(seg[:,:,:,0])
    
    
    ##### Tracking:
    print('\b\b\b - Tracking & Feature extraction ', end='')
    message = ''
    res = []
    for m, chamberbox in enumerate(chamberboxes):
        print('\b'*len(message),end='')
        message = 'chamber '+str(m+1)+'/'+str(len(chamberboxes))
        print(message,end='')
        
        track_inputs = [] # Here we work chamber by chamber for memory reasons
        frame_numbers = []# To reconstruct
        # Compile tracking inputs:
        for i in range(1,transframes.shape[0]):
            singleseg = getSinglecells(seg[m*transframes.shape[0] + i-1]) # Get single cell masks in the 'old' frame
            for c in range(singleseg.shape[0]):
                frame_numbers.append(i) # For lineage reconstruction later
                track_inputs.append(np.stack((
                        seg_inputs[m*transframes.shape[0] + i,:,:,0], # Current trans image
                        singleseg[c], # Mask of one previous cell
                        seg_inputs[m*transframes.shape[0] + i-1,:,:,0], # Previous trans image
                        seg[m*transframes.shape[0] + i]), # Mask of all current cells
                        axis=-1
                    ))
        track_inputs = np.array(track_inputs) # Format into 4D tensor
        # Run U-Net model for tracking:
        track = model_track.predict(track_inputs,verbose=0)
        
        
        ##### Compile lineage and label stack:
        label_stack = np.zeros([transframes.shape[0],seg.shape[1],seg.shape[2]],dtype=np.uint16)
        lineage, label_stack = updatelineage(seg[m*transframes.shape[0]], label_stack) # Initialize lineage and label stack on first frame
        for i in range(1,transframes.shape[0]):
            frame_idxs = [x for x, fn in enumerate(frame_numbers) if fn==i]
            if frame_idxs:
                scores = getTrackingScores(track_inputs[frame_idxs], track[frame_idxs])
                attrib = getAttributions(scores)
                lineage, label_stack = updatelineage(seg[m*transframes.shape[0] + i], label_stack, framenb=i, lineage=lineage, attrib=attrib) # Because we use uint16, we can only track up to 65535 cells per chamber
            
            
        ##### Extract features:
        # Add dictionary keys for the features:
        for l in range(len(lineage)):
            lineage[l] = {**lineage[l], **dict(length=[],width=[],area=[],pixels=[])}
            for f in range(fluoframes.shape[1]): #Go over all channels:
                lineage[l] = {**lineage[l], **{'fluo'+str(f+1):[]}}
        # Generate new label stack:
        label_stack_resized = np.empty([transframes.shape[0],chamberbox['ybr']-chamberbox['ytl'],chamberbox['xbr']-chamberbox['xtl']],dtype=np.uint16)
        for i in range(0,transframes.shape[0]): # Go over frames
            # Resize from original stack:
            label_stack_resized[i] = cv2.resize(label_stack[i],label_stack_resized.shape[2:0:-1],interpolation=cv2.INTER_NEAREST) # The stack needs to be resized to the actual chamber box size
            # Get cropped out fluorescence images:
            chamberfluo = np.empty((fluoframes.shape[1],)+label_stack_resized.shape[1:])
            for f in range(fluoframes.shape[1]):
                chamberfluo[f] = cropbox(fluoframes[i,f],chamberbox)
            # Get contours of all cells in frame:
            contours = cv2.findContours((label_stack_resized[i]>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] # Get contours of single cells
            contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sort along Y axis
            for c, contour in enumerate(contours): # Run through cells in frame
                cellnb = label_stack_resized[i,contour[0,0,1],contour[0,0,0]] # Get cell number in stack (opencv indexing is opposite of numpy)
                # Length, width, area:
                rotrect = cv2.minAreaRect(contour)
                lineage[cellnb-1]['length'].append(max(rotrect[1]))
                lineage[cellnb-1]['width'].append(min(rotrect[1]))
                lineage[cellnb-1]['area'].append(cv2.contourArea(contour))
                # Pixels list and Fluorescence:
                cellpixels = np.where(label_stack_resized[i]==cellnb)
                lineage[cellnb-1]['pixels'].append(np.ravel_multi_index(cellpixels,label_stack_resized.shape[1:]).astype(np.float32)) # Using floats for compatibility with previous version of the pipeline
                for f in range(fluoframes.shape[1]): #Go over all channels:
                    lineage[cellnb-1]['fluo'+str(f+1)].append(np.mean(chamberfluo[f,cellpixels[0],cellpixels[1]]))
        # Append to res dictionary:
        res.append(dict(lineage=lineage,label_stack=label_stack,label_stack_resized=label_stack_resized))
    
    # Save position .mat file:
    savemat(resfolder+'postprocessed/Position'+str(pos).zfill(6)+'.mat',{'res': res})
    print(' - Done! ['+str(round(time.time() - start_time))+' secs]')