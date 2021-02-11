"""
This script process an input file / folder with data from a multi-position 
microscopy experiment and produces .mat files describing cells lineage, 
fluorescence etc.

The script can be called from the command line:
    python pipeline.py /path/to/file/or/folder -o /path/to/output/folder
    
Or run from an IDE by changing file paths.

@author: jblugagne
"""

import config as cfg
# Modules:
import time, cv2, os, sys
import numpy as np
from data import postprocess
import utilities as utils


# Load models:
models = utils.loadmodels()

def process_xp(xpfolder, 
         prototype=None, 
         fileorder='pct',
         filenamesindexing=1,
         resfolder = None,
         rotation_correction = True,
         verbose = 1):
    '''
    Process entire experiment folder.

    Parameters
    ----------
    xpfolder : str
        Path to experiment file or folder.
    prototype : str, optional
        File names prototype. See xpreader() in utilities.py.
        The default is None.
    fileorder : str, optional
        File names order. See xpreader() in utilities.py. 
        The default is 'pct'.
    filenamesindexing : int, optional
        File names base indexing. See xpreader() in utilities.py.
        The default is 1.
    resfolder : str, optional
        Path to save results to. If None, a folder within the experiment folder
        will be created.
        The default is None.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    None.

    '''
    
    # Create folders if they don't exist:
    if resfolder is None:
        resfolder = os.path.join(xpfolder, 'delta_results')
    if not os.path.exists(resfolder):
        os.mkdir(resfolder)
    
    # Initialize XP reader:
    xp = utils.xpreader(xpfolder,
                        prototype=prototype,
                        fileorder=fileorder,
                        filenamesindexing=filenamesindexing)
    
    # Run over all positions:
    for pos in range(xp.positions):
        
        # Run processing for position:
        res, chamberboxes, rot_corr, driftvalues = process_position(xp,
                                                                    pos,
                                                                    verbose=verbose)

        # Save results to disk:
        utils.legacysave(os.path.join(resfolder,('Position%06d.mat' % (pos+1))), 
                         xpfolder, 
                         res, 
                         chamberboxes, 
                         rot_corr, 
                         driftvalues,
                         [xp.y,xp.x,xp.channels,xp.timepoints])
        
        
        
def process_position(xp,
                     pos,
                     rotation_correction = True,
                     verbose = 1):
    '''
    Apply DeLTA pipeline to a single position in the experiment

    Parameters
    ----------
    xp : xpreader object
        See xpreader() in utilities.py. 
    pos : int
        Position to process.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    res : list
        List of dictionaries containing the lineage and labels stacks.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    rot_corr : float
        Rotation applied to images to get the chambers horizontal.
    drift_values : list
        XY drift values computed by preprocessing step.

    '''
    
    ##### Preprocessing - Rotation correction, Chambers identificaton, Drift:
    start_time = time.time()
    if verbose: print('Pos. %d/%d: Preprocessing...' %(pos+1,xp.positions), end='')
    transframes, fluoframes, chamberboxes, rot_corr, driftvalues = preprocessing(xp, pos)
    if chamberboxes is None:
        if verbose: print('\b\b\b - No chambers detected! Aborting...')
        return None, None, rot_corr, None
    
    
    ##### Cell segmentation:
    if verbose: print('\b\b\b - Segmentation...', end='')
    seg, seg_inputs = segmentation(transframes, chamberboxes)
    
    
    ##### Tracking:
    res = trackinglineagefeatures(transframes, 
                                  fluoframes, 
                                  seg, 
                                  seg_inputs,
                                  chamberboxes)
    
    if verbose: print(' - Done! ['+str(round(time.time() - start_time))+' secs]')
    
    return res, chamberboxes, rot_corr, driftvalues

def preprocessing(xp, pos, rotation_correction = True):
    '''
    Rotation correction, drift correction, loading images...

    Parameters
    ----------
    xp : xpreader object
        See xpreader() in utilities.py. 
    pos : int
        Position to process.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.

    Returns
    -------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    rot_corr : float
        Rotation applied to images to get the chambers horizontal.
    drift_values : list
        XY drift values computed by preprocessing step.

    '''
    
    firstframe = xp.getframes(positions=pos,channels=0,frames=0,rescale=(0,1)) # Get first frame for rotation estimation (We assume first channel is trans)
    # Rotation correction:
    if rotation_correction:
        rot_corr = utils.deskew(firstframe) # Estimate rotation angle
    else:
        rot_corr = 0
    # Get trans frames:
    transframes = xp.getframes(positions=pos,channels=0,rescale=(0,1),rotate=rot_corr) # Now load all trans frames, with rotation correction
    firstframe = np.expand_dims(np.expand_dims(cv2.resize(transframes[0],(512,512)),axis=0),axis=3) # using expand_dims to get it into a shape that the chambers id unet accepts
    # Find chambers, filter results, get bounding boxes:
    chambermask = models['chambers'].predict(firstframe,verbose=0)
    chambermask = cv2.resize(np.squeeze(chambermask),transframes.shape[3:0:-1]) # scaling back to original size
    chambermask = postprocess(chambermask,min_size=cfg.min_chamber_area) # Binarization, cleaning and area filtering
    chamberboxes = utils.getChamberBoxes(np.squeeze(chambermask))
    # If no chambers detected:
    if len(chamberboxes)==0:
        return None, None, None, rot_corr, None
        
    # Drift correction:
    drifttemplate = utils.getDriftTemplate(chamberboxes, transframes[0]) # This template will be used as reference for drift correction
    driftcorbox = dict(xtl = 0,
                        xbr = None,
                        ytl = 0,
                        ybr = max(chamberboxes,key=lambda elem: elem['ytl'])['ytl']
                        ) # Box to match template
    transframes, driftvalues = utils.driftcorr(transframes, template=drifttemplate, box=driftcorbox) # Run drift corr    
    # Load up fluoresence images, apply drift correction and rotation:
    fluoframes = xp.getframes(positions=pos, channels=[x for x in range(1,xp.channels)], squeeze_dimensions=False, rotate=rot_corr)
    if fluoframes is not None:
        fluoframes = fluoframes[0] # Squeezing out the position dimension
        for f in range(fluoframes.shape[1]): #Go over all channels:
            fluoframes[:,f,:,:], _ = utils.driftcorr(fluoframes[:,f,:,:], drift=driftvalues) # Apply drift correction
    
    return transframes, fluoframes, chamberboxes, rot_corr, driftvalues

def segmentation(transframes, chamberboxes):
    '''
    

    Parameters
    ----------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.

    Returns
    -------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.

    '''
    
    seg_inputs = []
    # Compile segmentation inputs:
    for m, chamberbox in enumerate(chamberboxes):
        for i in range(transframes.shape[0]):
            seg_inputs.append(cv2.resize(utils.rangescale(utils.cropbox(transframes[i],chamberbox),(0,1)),(32,256)))
    seg_inputs = np.expand_dims(np.array(seg_inputs),axis=3) # Format into 4D tensor
    # Run segmentation U-Net:
    seg = models['segmentation'].predict(seg_inputs,verbose=0)
    seg = postprocess(seg[:,:,:,0])
    
    return seg, seg_inputs

def trackinglineagefeatures(transframes, 
                        fluoframes, 
                        seg, 
                        seg_inputs, 
                        chamberboxes,
                        verbose = 1):
    '''
    For each chamber, perform tracking, lineage reconstruction, and feature
    extraction.

    Parameters
    ----------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    res : list
        List of dictionaries containing the lineage and labels stacks.

    '''
    
    if verbose: 
        print('\b\b\b - Tracking & Feature extraction ', end='')
        message = ''
    res = []
    for m, chamberbox in enumerate(chamberboxes):
        if verbose: 
            print('\b'*len(message),end='')
            message = 'chamber '+str(m+1)+'/'+str(len(chamberboxes))
            print(message,end='')
        
        
        #### Tracking Unet:
        track, track_inputs, frame_numbers = tracking(seg, 
                                                      seg_inputs, 
                                                      m, 
                                                      transframes.shape[0])
        
        ##### Compile lineage and label stack:
        lin, label_stack = lineage(seg, 
                                   track, 
                                   track_inputs, 
                                   frame_numbers, 
                                   m, 
                                   transframes.shape[0])
            
        ##### Extract features:
        lin, label_stack_resized = features(lin,
                                            label_stack,
                                            fluoframes,
                                            chamberbox,
                                            transframes.shape[0])
        
            
        # Append to res dictionary:
        res.append(dict(lineage=lin,labelsstack=label_stack,labelsstack_resized=label_stack_resized))
    
    return res
    
def tracking(seg, seg_inputs, chamber_number, timepoints):
    '''
    Perform tracking for a single chamber.

    Parameters
    ----------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    chamber_number : int
        Number of the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    track : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 3.
    track_inputs : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 4.
    frame_numbers : list
        Frame number / timepoint of each tracking event.

    '''
    
    track_inputs = [] # Here we work chamber by chamber for memory reasons
    frame_numbers = []# To reconstruct
    # Compile tracking inputs:
    for i in range(1,timepoints):
        singleseg = utils.getSinglecells(seg[chamber_number*timepoints + i-1]) # Get single cell masks in the 'old' frame
        for c in range(singleseg.shape[0]):
            frame_numbers.append(i) # For lineage reconstruction later
            track_inputs.append(np.stack((
                    seg_inputs[chamber_number*timepoints + i,:,:,0], # Current trans image
                    singleseg[c], # Mask of one previous cell
                    seg_inputs[chamber_number*timepoints + i-1,:,:,0], # Previous trans image
                    seg[chamber_number*timepoints + i]), # Mask of all current cells
                    axis=-1
                ))
    track_inputs = np.array(track_inputs) # Format into 4D tensor
    # Run U-Net model for tracking:
    if track_inputs.size!=0: # If not empty
        track = models['tracking'].predict(track_inputs,verbose=0)
    
    return track, track_inputs, frame_numbers

def lineage(seg, track, track_inputs, frame_numbers, chamber_number, timepoints):
    '''
    Assemble lineage from tracking outputs.

    Parameters
    ----------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    track : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 3.
    track_inputs : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 4.
    frame_numbers : list
        Frame number / timepoint of each tracking event.
    chamber_number : int
        Number of the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber.
    label_stack : 3D array of uint16
        Images stack labelled with each cell in the chamber, over time. For 
        each frame in the stack, i-valued pixels correspond to the (i-1)-th 
        list element/cell in the lin output (0-based list indexing). The 
        dimensions are timepoints -by- 256 -by- 32.

    '''
    
    label_stack = np.zeros([timepoints,seg.shape[1],seg.shape[2]],dtype=np.uint16)
    lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints], label_stack) # Initialize lineage and label stack on first frame
    for i in range(1,timepoints):
        frame_idxs = [x for x, fn in enumerate(frame_numbers) if fn==i]
        if frame_idxs:
            scores = utils.getTrackingScores(track_inputs[frame_idxs[0],:,:,3], track[frame_idxs])
            attrib = utils.getAttributions(scores)
        else:
            attrib = []
        lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints + i], label_stack, framenb=i, lineage=lin, attrib=attrib) # Because we use uint16, we can only track up to 65535 cells per chamber
    
    return lin, label_stack

def features(lin, label_stack, fluoframes, chamberbox, timepoints):
    '''
    Extract cell features such as fluorescence, cell length etc...

    Parameters
    ----------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber.
    label_stack : 3D array of uint16
        Images stack labelled with each cell in the chamber, over time. For 
        each frame in the stack, i-valued pixels correspond to the (i-1)-th 
        list element/cell in the lin output (0-based list indexing).
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    chamberboxes : list
        Dictionary containing the position of the chamber crop box in the image
        for the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber (updated with single-cell features).
    label_stack_resized : 3D array of uint16.
        Resized version of label_stack, scaled back to the original chamber 
        images crop box size.

    '''
    
    # Add dictionary keys for the features:
    for l in range(len(lin)):
        lin[l] = {**lin[l], **dict(length=[],width=[],area=[],pixels=[])}
        if fluoframes is not None:
            for f in range(fluoframes.shape[1]): #Go over all channels:
                lin[l] = {**lin[l], **{'fluo'+str(f+1):[]}}
    # Generate new label stack:
    label_stack_resized = np.empty([timepoints,chamberbox['ybr']-chamberbox['ytl'],chamberbox['xbr']-chamberbox['xtl']],dtype=np.uint16)
    for i in range(0,timepoints): # Go over frames
        # Resize from original stack:
        label_stack_resized[i] = cv2.resize(label_stack[i],label_stack_resized.shape[2:0:-1],interpolation=cv2.INTER_NEAREST) # The stack needs to be resized to the actual chamber box size
        # Get cropped out fluorescence images:
        if fluoframes is not None:
            chamberfluo = np.empty((fluoframes.shape[1],)+label_stack_resized.shape[1:])
            for f in range(fluoframes.shape[1]):
                chamberfluo[f] = utils.cropbox(fluoframes[i,f],chamberbox)
        # Get contours of all cells in frame:
        contours = cv2.findContours((label_stack_resized[i]>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # Get contours of single cells
        contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sort along Y axis
        for c, contour in enumerate(contours): # Run through cells in frame
            cellnb = label_stack_resized[i,contour[0,0,1],contour[0,0,0]] # Get cell number in stack (opencv indexing is opposite of numpy)
            # Length, width, area:
            rotrect = cv2.minAreaRect(contour)
            lin[cellnb-1]['length'].append(max(rotrect[1]))
            lin[cellnb-1]['width'].append(min(rotrect[1]))
            lin[cellnb-1]['area'].append(cv2.contourArea(contour))
            # Pixels list and Fluorescence:
            cellpixels = np.where(label_stack_resized[i]==cellnb)
            lin[cellnb-1]['pixels'].append(np.ravel_multi_index(cellpixels,label_stack_resized.shape[1:]).astype(np.float32)) # Using floats for compatibility with previous version of the pipeline
            if fluoframes is not None:
                for f in range(fluoframes.shape[1]): #Go over all channels:
                    lin[cellnb-1]['fluo'+str(f+1)].append(np.mean(chamberfluo[f,cellpixels[0],cellpixels[1]]))
                        
    return lin, label_stack_resized

### Main
if __name__=='__main__':
    
    resfolder = None
    rotation_correction = True
    prototype = None
    filenamesindexing = 1
    fileorder = 'pct'
    verbose = 1
    
    if len(sys.argv) >= 3: # If arguments were passed
        xpfolder = sys.argv[1]
        resfolder = sys.argv[2]
        i = 3
        while i<len(sys.argv):
            if sys.argv[i]=='-order':
                fileorder = sys.argv[i+1]
                i+=2
            if sys.argv[i]=='-index':
                filenamesindexing = int(sys.argv[i+1])
                i+=2
            if sys.argv[i]=='-proto':
                prototype = sys.argv[i+1]
                i+=2
            if sys.argv[i]=='-rot':
                rotation_correction = float(sys.argv[i+1])
                i+=2
            if sys.argv[i]=='-v':
                verbose = int(sys.argv[i+1])
                i+=2
    
    else: # Interactive session:
        # Folder containing image sequence:
        xpfolder = utils.getxppathdialog()
        
        # Get xp settings:
        print('Experiment type?\n1 - bioformats2sequence\n2 - micromanager\n3 - high-throughput\n0 - other\nEnter a number [1]: ',end='')
        answer = int(input())
        print()
        if answer is None or answer == 1:
            prototype = None
            fileorder = 'pct'
            filenamesindexing=1
        elif answer == 2:
            prototype = 'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            fileorder = 'pcpt'
            filenamesindexing=0
        elif answer == 3:
            prototype = 'chan%02d_img/Position%06d_Frame%06d.tif'
            fileorder = 'cpt'
            filenamesindexing=1
        elif answer == 0:
            print('Enter files prototype: ', end='')
            prototype = input()
            print()
            print('Enter files order: ', end='')
            fileorder = input()
            print()
            print('Enter files indexing: ', end='')
            filenamesindexing = int(input())
            print()
        else:
            raise ValueError('Invalid experiment type')
        print()
    
    
    process_xp(xpfolder, 
              prototype=prototype,
              fileorder=fileorder,
              filenamesindexing=filenamesindexing,
              resfolder = resfolder,
              rotation_correction = rotation_correction,
              verbose = verbose)