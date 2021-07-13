'''
This file contains functions and class definitions that are used in pipeline.py
A lot of functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

@author: jblugagne
'''
import config as cfg
import cv2, os, re, ffmpeg, time
import numpy as np
from scipy.io import savemat
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from queue import Queue
from threading import Thread
from skimage.morphology import skeletonize


def deskew(image):
    '''
    Compute rotation angle of chambers in image for rotation correction.
    From: https://gist.github.com/panzerama/beebb12a1f9f61e1a7aa8233791bc253
    Not extensively tested. You can skip rotation correction if your chambers
    are about +/- 1 degrees of horizontal

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    rotation_number : float
        Rotation angle of the chambers for correction.

    '''
    
    from skimage.filters import gaussian, threshold_otsu
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line


    # canny edges in scikit-image
    edges = canny(image)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)
    
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number

def cropbox(img,box):
    '''
    Crop image

    Parameters
    ----------
    img : 2D numpy array
        Image to crop.
    box : Dictionary
        Dictionary describing the box to cut out, containing the following 
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    '''
    if box is None or all([v is None for v in box.values()]):
        return img
    else:
        return img[box['ytl']:box['ybr'],box['xtl']:box['xbr']]

def rangescale(frame, rescale):
    '''
    Rescale image values to be within range

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    '''
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame-np.min(frame))/np.ptp(frame))*np.ptp(rescale)+rescale[0]
    else:
        frame = np.ones_like(frame)*(rescale[0]+rescale[1])/2
    return frame

def imrotate(frame, rotate):
    '''
    Rotate image

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rotate : float
        Rotation angle, in degrees.

    Returns
    -------
    2D numpy array of floats
        Rotated image

    '''
    M = cv2.getRotationMatrix2D((frame.shape[1]/2,frame.shape[0]/2), rotate, 1)
    frame = cv2.warpAffine(frame,M,(frame.shape[1],frame.shape[0]),borderMode=cv2.BORDER_REPLICATE)
    
    return frame

def driftcorr(img,template=None,box=None, drift=None):
    '''
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drfit for.
    template : None or 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        default is None.
    box : None or dictionary, optional
        A cropping box to extract the part of the frame to compute drift 
        correction over (see cropbox()).
        default is None.
    drift : None or tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        default it None.

    Returns
    -------
    2D/3D numpy array, tuple of len 2
        Drift-corrected image and drift.

    '''
    
    if len(img.shape)==2:
        twoDflag=True
        img = np.expand_dims(img,axis=0)
    else:
        twoDflag=False
        
    if drift is None:
        if template is None: # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0,0)
        template = rangescale(template,(0,255)).astype(np.uint8) # Making sure its the right format
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]],[drift[1]])
    else:
        (xcorr, ycorr) = drift
        
    for i in range(img.shape[0]):
        if drift is None:
            frame = rangescale(img[i],(0,255)).astype(np.uint8) # Making sure its the right format
            driftcorrimg = cropbox(frame,box)
            res = cv2.matchTemplate(driftcorrimg,template,cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            ycorr[i] = max_loc[0] - res.shape[1]/2
            xcorr[i] = max_loc[1] - res.shape[0]/2
        T = np.float32([[1, 0, -ycorr[i]], [0, 1, -xcorr[i]]])
        img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])
    
    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    else:
        return img, (xcorr, ycorr)


def getROIBoxes(chambersmask):
    '''
    This function extracts the bounding boxes of the chambers in thebinary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of dictionaries
        List of cropping box dictionaries (see cropbox()).

    '''
    chamberboxes = []
    if chambersmask.dtype == bool:
        chambersmask = chambersmask.astype(np.uint8)
    else:
        chambersmask = cv2.threshold(chambersmask,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(chambersmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamberboxes.append(dict(
                xtl = xtl,
                ytl = int(ytl-(boxheight*.1)), # -10% of height to make sure the top isn't cropped
                xbr = xtl+boxwidth,
                ybr = ytl+boxheight)) # tl = top left, br = bottom right.
    chamberboxes.sort(key=lambda elem: elem['xtl'])# Sorting by top-left X (normally sorted by Y top left)
    return chamberboxes


def getDriftTemplate(chamberboxes,img,whole_frame=False):
    '''
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getROIBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.
    whole_frame : bool, optional
        Whether to use the whole frame as reference instead of the area above
        the chambers.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    '''
    
    if len(chamberboxes) == 0 and not whole_frame:
        return None
    (y_cut,x_cut) = [round(i*.025) for i in img.shape] # Cutting out 2.5% of the image on eahc side as drift margin
    
    box = dict(
        xtl = x_cut,
        xbr = -x_cut,
        ytl = y_cut,
        ybr = -y_cut if whole_frame else max(chamberboxes,
                                             key=lambda elem: elem['ytl']
                                             )['ytl']-y_cut
        )
        
    return cropbox(img,box)

def getSinglecells(seg):
    '''
    Return masks of single cells

    Parameters
    ----------
    seg : array of uint8/uint16/float/bool
        Mask of cells. Values >0.5 will be considered cell pixels

    Returns
    -------
    singlecellseg : 3D array of uint8
        Stack of single cell masks. Each single-cell mask is stacked along the
        first axis (axis 0)

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
    singlecellseg = np.empty([len(contours),seg.shape[0],seg.shape[1]])
    for c, contour in enumerate(contours):
        singlecellseg[c] = cv2.drawContours(np.zeros((seg.shape[0],seg.shape[1]), dtype=np.uint8),[contour],0, color=1, thickness=-1)
    return singlecellseg

def getTrackingScores(labels, outputs, boxes=None):
    '''
    Get overlap scores between input/target cells and tracking outputs

    Parameters
    ----------
    inputs : 2D array of floats
        Segmentation mask of input/target cells that the tracking U-Net is 
        tracking against. (ie segmentation mask of the 'current'/'new' frame)
    outputs : 4D array of floats
        Tracking U-Net output. For each cell in the 'old'/'previous' frame,
        two masks were produced as potential mother and daughter cell outlines.
        (plus a third, unused mask for background)

    Returns
    -------
    scores : 3D array of floats
        Overlap scores matrix between tracking predictions and current 
        segmentation mask for each new-old cell, either as a mother or a 
        daughter cell. Cells from the 'old' frame (axis 0) will get a tracking 
        score that corresponds to how much the U-Net tracking output mask 
        overlaps with the segmentation mask of each cell in the 'new' mask 
        (axis 1). Each old cell gets a tracking scores for each new cell as 
        either a potential daughter (mother-daughter relationship, index 1 on 
        axis 2) or as just itself (mother-mother relationship, index 0 on axis
        2)
        Example:
            attrib[1,3,0] == 1 means the 'mother' tracking output of cell #2 in
            the old frame completely overlaps with the mask of cell #4 in the 
            new frame.
            attrib[4,6,1] == 0.7 means the 'daughter' tracking output of cell
            #5 in the old frame overlaps with 70% of the mask of cell #7 in the
            new frame.

    '''

    total_cell = np.max(labels)
    if total_cell == 0:
        return None
    
    # Get areas for each cell:
    _, areas = np.unique(labels, return_counts=True)
    areas = areas[1:]
    
    # Compile scores:
    scores = np.zeros([outputs.shape[0],total_cell],dtype=np.float32)
    for o, output in enumerate(outputs):
        
        # Find pixels with a score > .05:
        nz = list((output>.05).nonzero())
        if len(nz[0])==0:
            continue
        
        # Find cells that are "hit" by those pixels in the labels image:
        if boxes is None or\
            boxes[o] is None or\
                all([v is None for v in boxes[o][0].values()]):
            cells, counts = np.unique(labels[nz], return_counts=True)
            
        else:
            # Clean nz hits outside of fillbox:
            cb, fb = boxes[o][:]
            to_remove = np.logical_or.reduce(
                (
                    nz[0]<=fb['ytl'],
                    nz[0]>=fb['ybr'],
                    nz[1]<=fb['xtl'],
                    nz[1]>=fb['xbr']
                    )
                )
            nz[0] = np.delete(nz[0],to_remove.nonzero())
            nz[1] = np.delete(nz[1],to_remove.nonzero())
            
            # Offset hits by cropbox-fillbox:
            nz[0] = nz[0]+cb['ytl']-fb['ytl']
            nz[1] = nz[1]+cb['xtl']-fb['xtl']
            
            # Compute number of hits per cell:
            cells, counts = np.unique(labels[nz[0],nz[1]], return_counts=True)
        
        # Compile score for these cells:
        for c, cell in enumerate(cells):
            if cell > 0:
                scores[o,cell-1]=counts[c]/areas[cell-1]
    
    return scores

        
def getOverlap(output,target):
    '''
    Get portion of tracking output overlapping on target cell

    Parameters
    ----------
    output : array of uint8
        Mask of tracking output.
    target : array of uint8
        Mask of target cell.

    Returns
    -------
    float
        Portion of target cell that is covered by the tracking output.

    '''
    return  cv2.sumElems(
                    cv2.bitwise_and(output,
                                    target)
                )[0]/\
            cv2.sumElems(
                    target
                )[0]

def getMult(output, target):
    '''
    Average tracking output score in target cell mask.

    Parameters
    ----------
    output : array of uint8
        Mask of tracking output.
    target : array of uint8
        Mask of target cell.

    Returns
    -------
    float
        Average tracking output score per target cell pixel.

    '''
    
    return np.sum(np.multiply(output,target))/np.sum(target)

def getAttributions(scores):
    '''
    Get attribution matrix from tracking scores

    Parameters
    ----------
    scores : numpy 3D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : numpy 3D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each new cell can be attributed to an
        old cell as a "mother", ie the same cell (index 0 along axis 2), or as
        a "daughter" of that old cell (index 1 along axis 2)
        Example:
            attrib[1,3,0] == True means cell #4 in the new frame (from top to 
            bottom) is attributed to cell #2 in the old frame.
            attrib[4,6,1] == True means cell #7 in the new frame is 
            attributed as a daughter to cell #5 in the old frame.
        Each 'old' cell can have only 0 or 1 'new' cell identified as itself 
        (ie mother-mother relationship), and 0 or 1 'new' cell identified as a
        daughter (ie mother-daughter relationship). Each 'new' cell can only be
        attributed to 0 or 1 'old' cell, either as a 'mother' or 'daughter'.

    '''
    
    attrib = np.zeros(scores.shape, dtype=int)
    
    for n in range(scores.shape[1]):
        worst_old = np.argsort(scores[:,n]) # Worst-to-best score of old cells for n-th new cell
        # Run through old cells from best to worst:
        for o in range(-1,-scores.shape[0]-1,-1):
            o_best = worst_old[o] # o-th best old cell 
            s = scores[o_best,n]
            # If score gets too low, stop
            if s < .2:
                break
            # Check if new cell is at least 2nd best for this old cell:
            worst_new = np.argsort(scores[o_best,:]) # Worst-to-best score of new cells for o_best old cell
            if n in worst_new[-2:]:
                attrib[o_best,n]=1
                break
                

    return attrib

def label_seg(seg,cellnumbers=None,get_contours=False,background=0):
    '''
    Label cells in segmentation mask

    Parameters
    ----------
    seg : numpy 2D array of float/uint8/uint16/bool
        Cells segmentation mask. Values >0.5 will be considered cell pixels
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is 
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    label : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels 
        with values given by cellnumbers

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
    label = np.full(seg.shape,background,dtype=np.uint16)
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(label,[contour],c+1 if cellnumbers is None else cellnumbers[c])
    if get_contours:
        return label, contours
    else:
        return label

def updatelineage(newframe, stack, framenb=0, lineage=None, attrib=None):
    '''
    Update lineage list and labels stack based on attribution matrix

    Parameters
    ----------
    newframe : numpy 2D array
        Segmentation output for current frame.
    stack : numpy 3D tensor of uint16
        Labelled stack. Each frame until the current frame contains a numbered 
        mask, with the number corresponding the cell number in the lineage. See
        label_seg()
    framenb : int, optional
        Current frame number (0-based indexing).
        The default is 0.
    lineage : List of dicts, optional
        List of all the cells encountered in the chamber. Each cell dict 
        contains the following keys:
            - mother, int: The number of the mother cell in the lineage 
            (1-based indexing)
            - framenbs, list of ints: The list of frames where the cell is 
            present (1-based indexing)
            - daughters, list of ints: A list of the same length as framenbs,
            with each element equal to the number of the daughter in the 
            lineage (1-based indexing) if a division occured at the 
            corresponding frame number, and equal to 0 otherwise.
        Note: We used 1-based indexing for compatibility with earlier code that
        was written in Matlab.
        If None is provided, the lineage will be initialized and all cells in 
        the newframe image will be given 0 as a mother.
        The default is None.
    attrib : numpy 2D array of bool, optional
        Attribution matrix for the current frame/timepoint as produced by 
        number 0. If None, lineage and stack will not be updated.
        getAttributions(). New cells with no attributions are given mother cell
        The default is None.

    Returns
    -------
    lineage : List of dicts
        lineage updated for the current frame and attrib matrix. See lineage 
        input description above.
    stack : numpy 3D tensor of uint16
        labelled stack updated for the current frame and attrib matrix. See 
        lineage input description above..

    '''
    
    if lineage==None: # Not initialized yet
        lineage = []
        stack[framenb] = label_seg(newframe)
        cells = np.unique(stack[framenb])
        for c in cells[1:]: # Avoiding cell==0 (background)
            lineage.append(createcell(framenb+1)) # We're using framenb +1 to be compatible with old Matlab code
    else: # Update according to attribution matrix
        # Init cell numbers, retrieve previous stack frame:
        cellnumbers = []
        label_old = label_seg((stack[framenb-1]>0).astype(np.uint8))
        
        # If previous stack frame has no cells:
        if not np.any(label_old>0): # This can happen if the chamber is empty and suddenly cells appear (can be a segmentation error, or a focus problem)
            contours = cv2.findContours(newframe.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # In this case just label
            for _ in range(len(contours)):
                lineage.append(createcell(framenb+1))
                cellnumbers.append(len(lineage))
                
        # If previous stack frame has cells:
        else:
            for i in range(attrib.shape[1]): # Go through new cells
                if np.any(attrib[:,i,0]): # If cell is tracked from an old cell: (mother-mother)
                    cellnum = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,0])[0]+1)[0]][0]) # retrieving "mother" cell number from old stack frame
                    lineage[cellnum-1]['framenbs'].append(float(framenb+1)) # Update frame number list
                    if len(lineage[cellnum-1]['daughters']) < len(lineage[cellnum-1]['framenbs']): # If framenbs hasn't been updated (in case somehow daughter goes through before mother)
                        lineage[cellnum-1]['daughters'].append(0.)
                    cellnumbers.append(cellnum)
                elif np.any(attrib[:,i,1]): # Cell is tracked as a daughter cell (mother-daughter)
                    mothernb = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,1])[0]+1)[0]][0]) # retrieving cell number from stack
                    # Create new cell, as daughter of mother:
                    lineage.append(createcell(framenb+1,mothernb))
                    cellnumbers.append(len(lineage))
                    # Update 'daughters' key of mother cell:
                    if len(lineage[mothernb-1]['daughters']) < len(lineage[mothernb-1]['framenbs']): # This situation should never happen actually
                        lineage[mothernb-1]['daughters'].append(float(cellnumbers[-1]))
                    else:
                        lineage[mothernb-1]['daughters'][-1] = float(cellnumbers[-1])
                else: # Orphan cell
                    lineage.append(createcell(framenb+1))
                    cellnumbers.append(len(lineage))
                    
        # Compile new labeled frame, push to stack:
        stack[framenb] = label_seg(newframe,cellnumbers=cellnumbers)
    return lineage, stack
        

def createcell(framenb, mothernb=0):
    '''
    Create cell to append to lineage list

    Parameters
    ----------
    framenb : int
        Frame that the cell first appears (1-based indexing).
    mothernb : int, optional
        Number of the mother cell in the lineage (1-based indexing).
        The default is 0. (ie unknown mother)

    Returns
    -------
    dict
        Initialized cell dictionary.

    '''
    return dict(framenbs=[float(framenb)], # For compatibility with old version of the pipeline
                mothernb=mothernb,
                daughters=[0.])

def loadmodels(toload=cfg.models):
    '''
    Load models (as specified in config.py)

    Parameters
    ----------
    toload : list of str, optional
        Which of the 3 models to load. 
        The default is ['rois','segmentation','tracking'].

    Returns
    -------
    models : dict
        Dictionary containing the models specified.

    '''
    
    
    from model import unet_rois, unet_seg, unet_track
    models = dict()
    
    if 'rois' in toload:
        models['rois'] = unet_rois(input_size = cfg.target_size_rois + (1,))
        models['rois'].load_weights(cfg.model_file_rois)
    
    if 'segmentation' in toload:
        models['segmentation'] = unet_seg(input_size = cfg.target_size_seg + (1,))
        models['segmentation'].load_weights(cfg.model_file_seg)
    
    if 'tracking' in toload:
        models['tracking'] = unet_track(input_size = cfg.target_size_track + (4,))
        models['tracking'].load_weights(cfg.model_file_track)
    
    return models

def legacysave(res_file, 
               xpfile, 
               delta_res, 
               chamberboxes, 
               rot_corr, 
               driftvalues,
               moviedimensions):
    '''
    Save single position pipeline results in the legacy Matlab format

    Parameters
    ----------
    res_file : str
        Path to save the data to.
    xpfile : str
        Path to the experimental folder.
    res : list
        List of dicts for each chamber produced by delta pipeline.
    chamberboxes : list
        List of dicts describing single chambers positions in the image.
    rot_corr : float
        Rotation applied to images to get the chambers horizontal.
    driftvalues : list
        XY drift values computed by pipeline preprocessing step.

    Returns
    -------
    None.

    '''
    
    if delta_res is not None:
        savemat(res_file,
                {'res': delta_res,
                 'tiffile': xpfile,
                 'moviedimensions': moviedimensions,
                 'proc': {'rotation': rot_corr,
                          'chambers': np.array([[c['xtl'],c['ytl'],c['xbr']-c['xtl'],c['ybr']-c['ytl']] for c in chamberboxes],dtype=np.float64),
                          'XYdrift': np.flip(np.transpose(-np.array(driftvalues)),axis=1)}
                 })
    
    else:
        savemat(res_file,
                {'res': [],
                 'tiffile': xpfile,
                 'moviedimensions': moviedimensions,
                 'proc': {'rotation': rot_corr,
                          'chambers': [],
                          'XYdrift': []}
                 })

#TODO
def getxppathdialog(ask_folder=False):
    '''
    Pop up window to select experiment file or folder.

    Parameters
    ----------
    ask_folder : bool, optional
        Folder selection window will pop up instead of file selection.
        The default is False.

    Returns
    -------
    file_path : str
        Path to experiment file or folder.

    '''
    
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    # root.withdraw() # For some reason this doesn't work with askdirectory?
    if ask_folder:
        file_path = filedialog.askdirectory(title='Please select experiment folder(s)',
                                              mustexist=True)
        
    else:
        file_path = filedialog.askopenfilenames(title='Please select experiment files(s)')
    
    root.destroy()
        
    return file_path


def getcellsinframe(labels, get_contours=False, background=0):
    '''
    Get numbers of cells present in frame, sorted along Y axis

    Parameters
    ----------
    labels : 2D numpy array of ints
        Single frame from labels stack.
    get_contours : bool, optional
        Flag to get cv2 contours.

    Returns
    -------
    cells : list
        Cell numbers (0-based indexing).
    contours : list
        List of cv2 contours for each cell. Returned if get_contours==True.

    '''
    
    cells, ind = np.unique(labels,return_index=True)
    cells = [cell-1 for _,cell in sorted(zip(ind,cells))] # Sorting along Y axis, 1-based to 0-based, & removing 1st value (background)
    cells = [cell for cell in cells if cell!=background-1] # Remove background
    
    # Get opencv contours:
    if get_contours:
        contours = []
        for c, cell in enumerate(cells):
            # Have to do it this way to avoid cases where border is shrunk out
            cnt, _ = cv2.findContours(
                (labels==cell+1).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
                )
            contours += cnt
        return cells, contours
    else:
        return cells

def results_movie(position, frames=None):
        
    
        # Re-read trans frames:
        trans_frames = position.reader.getframes(
            positions=position.position_nb,
            channels=0,
            frames=frames,
            rescale=(0, 1),
            squeeze_dimensions=False,
            rotate=position.rotate
            )
        trans_frames = trans_frames[0,:,0]
        if position.drift_correction:
            trans_frames, _ = driftcorr(
                trans_frames, drift=position.drift_values
                )
        movie = []
        
        # Run through frames, compile movie:
        for f, fnb in enumerate(frames):
            
            frame = trans_frames[f]
            
            #RGB-ify:
            frame = np.repeat(frame[:,:,np.newaxis],3,axis=-1)
            
            # Add frame number text:
            frame = cv2.putText(
                frame,
                text='frame %06d' %(fnb,),
                org=(int(frame.shape[0]*.05),int(frame.shape[0]*.97)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 1, 1, 1),
                thickness=2
                )
            
            for r, roi in enumerate(position.rois):
                    
                    # Get chamber-specific variables:
                    colors = getrandomcolors(len(roi.lineage.cells), seed=r)
                    cells, contours = getcellsinframe(
                        roi.label_stack[fnb],
                        get_contours=True
                        )
                    
                    if roi.box is None:
                        xtl, ytl = (0, 0)
                    else:
                        xtl, ytl = (roi.box['xtl'], roi.box['ytl'])
                    
                    # Run through cells in labelled frame:
                    for c, cell in enumerate(cells):
                        
                        # Draw contours:
                        frame = cv2.drawContours(
                            frame,
                            contours,
                            c,
                            color=colors[cell],
                            thickness=1,
                            offset=(xtl,ytl)
                            )
                        
                        # Draw poles:
                        oldpole = roi.lineage.getvalue(cell,'old_pole',fnb)
                        frame = cv2.drawMarker(
                            frame, 
                            tuple(oldpole[::-1]), 
                            color=colors[cell],
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=3,
                            thickness=1
                            )
                        
                        daughter = roi.lineage.getvalue(cell,'daughters',fnb)
                        bornago = roi.lineage.cells[cell]['frames'].index(fnb)
                        
                        if daughter is None and bornago>0:
                            newpole = roi.lineage.getvalue(cell,'new_pole',fnb)
                            frame = cv2.drawMarker(
                                frame,
                                tuple(newpole[::-1]),
                                color=[1,1,1], 
                                markerType=cv2.MARKER_TILTED_CROSS,
                                markerSize=3,
                                thickness=1
                                )
                        
                        # Plot division arrow:
                        if daughter is not None:
                            
                            newpole = roi.lineage.getvalue(cell,'new_pole',fnb)
                            daupole = roi.lineage.getvalue(daughter,'new_pole',fnb)
                            # Plot arrow:
                            frame = cv2.arrowedLine(
                                frame,
                                tuple(newpole[::-1]),
                                tuple(daupole[::-1]),
                                color=(1,1,1),
                                thickness=1
                                )
            
            # Add to movie array:
            movie+= [(frame*255).astype(np.uint8)]
        
        return movie

def getrandomcolors(num_colors, seed=0):
    '''
    Pseudo-randomly generate list of random hsv colors.

    Parameters
    ----------
    num_colors : int
        Number of desired colors.
    seed : None or int, optional
        Seed to use with numpy.random.seed(). 
        The default is 0.

    Returns
    -------
    colors : list
        List of RGB values (0-1 interval).

    '''
    
    # Get colors:
    cmap = cm.get_cmap('hsv', lut=num_colors)
    colors = [cmap(i)[0:3] for i in range(num_colors)]
    
    # Pseudo randomly shuffle colors:
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(colors)
    
    return colors

def getcentroid(contour):
    '''
    Get centroid of cv2 contour

    Parameters
    ----------
    contour : 3D numpy array
        Blob contour generated by cv2.findContours().

    Returns
    -------
    cx : int
        X-axis coordinate of centroid.
    cy : int
        Y-axis coordinate of centroid.

    '''
    
    if contour.shape[0]>2: # Looks like cv2.moments treats it as an image
        # Calculate moments for each contour
        M = cv2.moments(contour)
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
    else:
        cx = int(np.mean(contour[:,:,0]))
        cy = int(np.mean(contour[:,:,1]))
        
    return cx,cy



def vidwrite(images, filename, crf=20, verbose=1):
    '''
    Write images stack to video file with h264 compression.

    Parameters
    ----------
    images : 4D numpy array
        Stack of RGB images to write to video file.
    filename : str
        File name to write video to. (Overwritten if exists)
    crf : int, optional
        Compression rate. 'Sane' values are 17-28. See 
        https://trac.ffmpeg.org/wiki/Encode/H.264
        The default is 23.
    verbose : int, optional
        Verbosity of console output. 
        The default is 1.

    Returns
    -------
    None.

    '''
    
    # Initialize ffmpeg parameters:
    height,width,_ = images[0].shape
    quiet = [] if verbose else ['-loglevel', 'error', '-hide_banner']
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height),r=7)
            .output(filename, pix_fmt='yuv420p', vcodec='libx264', crf=crf, preset='veryslow')
            .global_args(*quiet)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    
    # Write frames:
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    
    # Close file stream:
    process.stdin.close()
    
    # Wait for processing + close to complete:
    process.wait()


def updatelineage(newframe, stack, framenb=0, lineage=None, attrib=None):
    '''
    Update lineage list and labels stack based on attribution matrix

    Parameters
    ----------
    newframe : numpy 2D array
        Segmentation output for current frame.
    stack : numpy 3D tensor of uint16
        Labelled stack. Each frame until the current frame contains a numbered 
        mask, with the number corresponding the cell number in the lineage. See
        label_seg()
    framenb : int, optional
        Current frame number (0-based indexing).
        The default is 0.
    lineage : List of dicts, optional
        List of all the cells encountered in the chamber. Each cell dict 
        contains the following keys:
            - mother, int: The number of the mother cell in the lineage 
            (1-based indexing)
            - framenbs, list of ints: The list of frames where the cell is 
            present (1-based indexing)
            - daughters, list of ints: A list of the same length as framenbs,
            with each element equal to the number of the daughter in the 
            lineage (1-based indexing) if a division occured at the 
            corresponding frame number, and equal to 0 otherwise.
        Note: We used 1-based indexing for compatibility with earlier code that
        was written in Matlab.
        If None is provided, the lineage will be initialized and all cells in 
        the newframe image will be given 0 as a mother.
        The default is None.
    attrib : numpy 2D array of bool, optional
        Attribution matrix for the current frame/timepoint as produced by 
        number 0. If None, lineage and stack will not be updated.
        getAttributions(). New cells with no attributions are given mother cell
        The default is None.

    Returns
    -------
    lineage : List of dicts
        lineage updated for the current frame and attrib matrix. See lineage 
        input description above.
    stack : numpy 3D tensor of uint16
        labelled stack updated for the current frame and attrib matrix. See 
        lineage input description above..

    '''
    
    if lineage==None: # Not initialized yet
        lineage = []
        stack[framenb] = label_seg(newframe)
        cells = np.unique(stack[framenb])
        for c in cells[1:]: # Avoiding cell==0 (background)
            lineage.append(createcell(framenb+1)) # We're using framenb +1 to be compatible with old Matlab code
    else: # Update according to attribution matrix
        # Init cell numbers, retrieve previous stack frame:
        cellnumbers = []
        label_old = label_seg((stack[framenb-1]>0).astype(np.uint8))
        
        # If previous stack frame has no cells:
        if not np.any(label_old>0): # This can happen if the chamber is empty and suddenly cells appear (can be a segmentation error, or a focus problem)
            contours = cv2.findContours(newframe.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # In this case just label
            for _ in range(len(contours)):
                lineage.append(createcell(framenb+1))
                cellnumbers.append(len(lineage))
                
        # If previous stack frame has cells:
        else:
            for i in range(attrib.shape[1]): # Go through new cells
                if np.any(attrib[:,i,0]): # If cell is tracked from an old cell: (mother-mother)
                    cellnum = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,0])[0]+1)[0]][0]) # retrieving "mother" cell number from old stack frame
                    lineage[cellnum-1]['framenbs'].append(float(framenb+1)) # Update frame number list
                    if len(lineage[cellnum-1]['daughters']) < len(lineage[cellnum-1]['framenbs']): # If framenbs hasn't been updated (in case somehow daughter goes through before mother)
                        lineage[cellnum-1]['daughters'].append(0.)
                    cellnumbers.append(cellnum)
                elif np.any(attrib[:,i,1]): # Cell is tracked as a daughter cell (mother-daughter)
                    mothernb = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,1])[0]+1)[0]][0]) # retrieving cell number from stack
                    # Create new cell, as daughter of mother:
                    lineage.append(createcell(framenb+1,mothernb))
                    cellnumbers.append(len(lineage))
                    # Update 'daughters' key of mother cell:
                    if len(lineage[mothernb-1]['daughters']) < len(lineage[mothernb-1]['framenbs']): # This situation should never happen actually
                        lineage[mothernb-1]['daughters'].append(float(cellnumbers[-1]))
                    else:
                        lineage[mothernb-1]['daughters'][-1] = float(cellnumbers[-1])
                else: # Orphan cell
                    lineage.append(createcell(framenb+1))
                    cellnumbers.append(len(lineage))
                    
        # Compile new labeled frame, push to stack:
        stack[framenb] = label_seg(newframe,cellnumbers=cellnumbers)
    return lineage, stack

def findfirst(mylist):
    return next((i for i, x in enumerate(mylist) if x>0), None)

def load_position(filename):
    
    import pickle
    f = open(filename, 'rb')
    p = pickle.load(f)
    f.close()
    return p


def stitch_pic(results,loc_y,loc_x,target_size = (512,512),overlap = False):
    
    '''

    Parameters
    ----------
    results : 4D numpy array
        Segmentation outputs from the seg model with dimensions #_of_windows -by- target_size[0] -by- target_size[1] -by- 1
        results contain all cropped segmentations from one image.
    loc_y_slice : numpy array with size = (2, number of windows)
        Give the location of the slices in axis = 0. (Split the image vertically)
    loc_x_slice : numpy array with size = (2, number of windows)
        Give the location of teh slices in axis = 1. (Split the image horizontally)
    target_size : tuple of 2 int, optional
        target_size is used to reconstruct the segmented image.
        The default is (512,512).
    overlap : bool, optional
        If True, when segmentations are overlapping, the average will be taken.
        If False, segmentations will be cropped so none are overlapping.
        The default is False.
        False is recommened as the model does not do as well near the edges 

    Returns
    -------
    stitch_norm : numpy array 3D with shape (original image shape + (1,))
        returns the stitched segmented image as a 3D numpy array with values between 0 and 1.

    '''
    
    # Create an array to store segmentations into a format similar to how the image was cropped
    stitch = np.zeros((loc_y[-1][1],loc_x[-1][1]),dtype=results.dtype)
    index = 0
    y_end = 0
    for i in range(len(loc_y)):
        
        # Compute y location of window:
        y_start = y_end
        if i+1==len(loc_y):
            y_end = loc_y[i][1]
        else:
            y_end = int((loc_y[i][1]+loc_y[i+1][0])/2)
        
        x_end = 0
        for j in range(len(loc_x)):
            
            # Compute x location of window:
            x_start = x_end
            if j+1==len(loc_x):
                x_end = loc_x[j][1]
            else:
                x_end = int((loc_x[j][1]+loc_x[j+1][0])/2)
            
            # Add to array:
            res_crop_y = -(loc_y[i][1]-y_end) if loc_y[i][1]-y_end>0 else None
            res_crop_x = -(loc_x[j][1]-x_end) if loc_x[j][1]-x_end>0 else None
            stitch[y_start:y_end,x_start:x_end] = results[
                index,
                y_start-loc_y[i][0]:res_crop_y,
                x_start-loc_x[j][0]:res_crop_x
                ]
            
            index+=1

    return stitch


def create_windows(img, target_size = (512,512), min_overlap = 24):
    '''
    The model only takes one input size.
    This function gets the location of indeces of how the image will split mulitple windows.
    Then it will crop the image into multiple windows based on these indeces

    Parameters
    ----------
    img : 2D or 3D numpy array
        For segmentation, the model only accepts images with size (512,512). 
        The image needs to be split into windows of size (512,512) that cover the whole image.
    target_size : tuple, optional
        The target size is the size of the windows which wil be fed into the mdoel.
        The default is (512,512).
    overlap : int, optional
        Overlap determines how much overlap between windows.
        Defaul is 24 pixels.
        

    Returns
    -------
    
    windows: numpy arr with size (# of slices in the y direction, # of slices in the x direction, img.shape[0], img.shape[1], 1)
        Gives the image cropped into the target size needed to feed the model
    loc_y_slice : numpy array with size = (2, number of windows)
        Give the location of the slices in axis = 0. (Split the image vertically)
    loc_x_slice : numpy array with size = (2, number of windows)
        Give the location of teh slices in axis = 1. (Split the image horizontally)

    Window n is:
        img[loc_y_slice[n,0] : loc_y_slice[n,1], loc_x_slice[n,0] : loc_x_slice[n,1]]

    The loc_y_slice & loc_x_slice are needed as outputs to stitch the segmented images back together

    '''
            
    # Decides how many images vertically the image is split into
    ny = int(
        1+float(img.shape[0]-min_overlap)/float(target_size[0]-min_overlap)
        )
    nx = int(
        1+float(img.shape[1]-min_overlap)/float(target_size[1]-min_overlap)
        )
    
    # Compute y-axis indices:
    ovlp_y = -(img.shape[0]-ny*target_size[0])/(ny-1)
    loc_y = []
    for n in range(ny-1):
        loc_y += [
            (int(target_size[0]*n-ovlp_y*n),int(target_size[0]*(n+1)-ovlp_y*n))
            ]
    loc_y += [(img.shape[0]-target_size[0],img.shape[0])]
    
    # Compute x-axis indices:
    ovlp_x = -(img.shape[1]-nx*target_size[1])/(nx-1)
    loc_x = []
    for n in range(nx-1):
        loc_x += [
            (int(target_size[1]*n-ovlp_x*n),int(target_size[1]*(n+1)-ovlp_x*n))
            ]
    loc_x += [(img.shape[1]-target_size[1],img.shape[1])]
    
    # Store all cropped images into one numpy array called windows
    windows = np.zeros(((nx*ny,)+target_size),dtype=img.dtype)
    for i in range(len(loc_y)):
        for j in range(len(loc_x)):
            windows[i*len(loc_x)+j,:,:] = img[
                loc_y[i][0]:loc_y[i][1],
                loc_x[j][0]:loc_x[j][1]
                ]

    return windows, loc_y, loc_x

def gettrackingboxes(cell, frame_shape, target_size=cfg.target_size_track):
    
    cx, cy = getcentroid(cell)
    
    xtl = int(max(cx-target_size[1]/2,0))
    xbr = int(min(cx+target_size[1]/2,frame_shape[1]))
    
    ytl = int(max(cy-target_size[0]/2,0))
    ybr = int(min(cy+target_size[0]/2,frame_shape[0]))
    
    cropbox = {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    xtl = int(max(target_size[1]/2-cx,0))
    xbr = int(min(target_size[1]/2+frame_shape[1]-cx,target_size[1]))
    
    ytl = int(max(target_size[0]/2-cy,0))
    ybr = int(min(target_size[0]/2+frame_shape[0]-cy,target_size[0]))
    
    fillbox = {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    return cropbox, fillbox


def getpoles(seg, labels=None):
    
    if labels is None:
        labels = label_seg(seg)
    skel = skeletonize(seg, method='lee')
    ends = skeleton_poles(skel)
    poles = extract_poles(ends, labels)
    for p in range(len(poles)):
        if len(poles[p])>2:
            poles[p] = check_two_endpoints(poles[p])
        if len(poles[p])<2: # Sometimes skeletonize fails
            poles[p] = extrema_poles(labels==p+1)
    
    return poles

def skeleton_poles(skel):
    """
    This function was adapted from stackoverflow
    #https://stackoverflow.com/questions/26537313/how-can-i-find-endpoints-of-binary-skeleton-image-in-opencv

    It uses a kernel to filter out the endpoints

    Parameters
    ----------
    skel : 2D numpy array of bool
        Contains skeletons of single cells from the segmentation

    Returns
    -------
    out : 2D numpy array 
        Contains the endpoints of the skeletons from the input (skel).
        
    """

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 1
    return out

def extrema_poles(cellmask):
    
    y,x = cellmask.nonzero()
    if np.ptp(y)>np.ptp(x): # Vertical cell
        i = np.lexsort((x,y))
    else: # Horizontal cell
        i = np.lexsort((y,x))    
    poles = [np.array((y[i[0]],x[i[0]])),np.array((y[i[-1]],x[i[-1]]))]
    
    return poles

def check_two_endpoints(endpoints):
    '''
    
    Sometimes the skeleton of a segmentation of a single cell will give multiple endpoints
    This function takes one of the endpoints at random and finds the other endpoint with the longest distance. Then it will find the endpoint with the longest distance from that endpoint
    This function will do nothing if the number of endpoints are 2    

    Parameters
    ----------
    endpoints : 2D numpy array
        Coordinates for the mother/daughter poles for a single cell
        Dimension is #_of_endpoints -by- [y_coordiante x_coordinate] (2) 
        

    Returns
    -------
    endpoints : 2D numpy array
        Dimensions are 2 -by- [y_coordiante x_coordinate] (2). It is not known which pole is the mother/daughter pole

    '''

    dist = 0
    # Measure the distance from the first endpoint to all the other endpoints
    for i in range(1,len(endpoints)):
        d = eucl(endpoints[0],endpoints[i])
        if d > dist:
            dist = d
            i_max = i
    
    # Redo the same thing starting from the furthest endpoint:
    dist = 0
    for j in range(len(endpoints)):
        if j != i_max: # skip same point
            d = eucl(endpoints[i_max],endpoints[j])
            if d > dist:
                dist = d
                j_max = j

    return [endpoints[i_max],endpoints[j_max]]
    
def extract_poles(end_img, labels):
    
    locations = end_img.nonzero()
    poles = [[] for _ in range(np.max(labels))]
    for p in range(len(locations[0])):
        
        cell = labels[locations[0][p],locations[1][p]]-1
        poles[cell]+=[np.array([locations[0][p],locations[1][p]])]
    
    return poles

def eucl(p1,p2):
    return np.linalg.norm(p1-p2)

class Lineage:
    
    def __init__(self):
        self.cells = []
        self.stack = []
        self.cellnumbers = []
    
    def update(self, cell, frame, attrib=[], poles=[]):
        
        # Check cell numbers list:
        if len(self.cellnumbers)<=frame:
            self.cellnumbers+=[[] for _ in range(frame+1-len(self.cellnumbers))] # Initialize
        if len(attrib)>0 and len(self.cellnumbers[frame])<=max(attrib):
            self.cellnumbers[frame]+=[-1 for _ in range(max(attrib)+1-len(self.cellnumbers[frame]))] # Initialize
        
        # If no attribution: (new/orphan cell)
        if cell is None and len(attrib)>0:
            cellnum = self.createcell(frame,new_pole=poles[0][0],old_pole=poles[0][1])
            self.cellnumbers[frame][attrib[0]]=cellnum
        
        # Simple tracking event:
        elif len(attrib)==1:
            cellnum = self.cellnumbers[frame-1][cell] # Get old cell number
            self.cellnumbers[frame][attrib[0]]=cellnum
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(cellnum, 'old_pole', frame-1)
            prev_new = self.getvalue(cellnum, 'new_pole', frame-1)
            # Find which pole is new vs old:
            if eucl(poles[0][0],prev_old)+eucl(poles[0][1],prev_new)\
                < eucl(poles[0][0],prev_new)+eucl(poles[0][1],prev_old):
                self.updatecell(
                    cellnum,frame,old_pole=poles[0][0],new_pole=poles[0][1]
                    )
            else:
                self.updatecell(
                    cellnum,frame,old_pole=poles[0][1],new_pole=poles[0][0]
                    )
            
        
        # Division event:
        elif len(attrib)==2:
            mothernum = self.cellnumbers[frame-1][cell] # Get old cell number
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(mothernum, 'old_pole', frame-1)
            prev_new = self.getvalue(mothernum, 'new_pole', frame-1)
            # Find new new poles (2 closest of the poles of the new cells):
            min_dist = np.inf
            for c1 in range(2):
                for c2 in range(2):
                    dist = eucl(poles[0][c1],poles[1][c2])
                    if dist<min_dist:
                        min_dist=dist
                        c1_new = c1
                        c2_new = c2
            # Find poles closest to old and new pole from previous cell:
            if eucl(poles[0][int(not c1_new)],prev_old) \
                + eucl(poles[1][int(not c2_new)],prev_new) \
                    < eucl(poles[0][int(not c1_new)],prev_new) \
                        + eucl(poles[1][int(not c2_new)],prev_old):
                # cell 0 is mother, cell 1 is daughter
                # Attribute daughter:
                daughternum = self.createcell(
                    frame,
                    new_pole=poles[1][c2_new],
                    old_pole=poles[1][int(not c2_new)],
                    mother=mothernum
                    )
                self.cellnumbers[frame][attrib[1]]=daughternum
                #Attribute mother:
                self.updatecell(
                    mothernum,
                    frame,
                    new_pole=poles[0][c1_new],
                    old_pole=poles[0][int(not c1_new)],
                    daughter=daughternum
                    )
                self.cellnumbers[frame][attrib[0]]=mothernum
            else:
                # cell 1 is mother, cell 0 is daughter
                # Attribute daughter:
                daughternum = self.createcell(
                    frame,
                    new_pole=poles[0][c1_new],
                    old_pole=poles[0][int(not c1_new)],
                    mother=mothernum
                    )
                self.cellnumbers[frame][attrib[0]]=daughternum
                #Attribute mother:
                self.updatecell(
                    mothernum,
                    frame, 
                    new_pole=poles[1][c2_new], 
                    old_pole=poles[1][int(not c2_new)],
                    daughter=daughternum
                    )
                self.cellnumbers[frame][attrib[1]]=mothernum
            
            if daughternum not in self.cellnumbers[frame]:
                raise RuntimeError('daughter not added')
            
    def createcell(self, frame, new_pole=None, old_pole=None, mother=None):
        '''
        Create cell to append to lineage list
    
        Parameters
        ----------
        frame : int
            Frame that the cell first appears (1-based indexing).
        mother : int, optional
            Number of the mother cell in the lineage (1-based indexing).
            The default is 0. (ie unknown mother)
    
        Returns
        -------
        int
            new cell number
    
        '''
        
        new_cell = {
            'id':len(self.cells),
            'mother':mother,
            'frames':[frame],
            'daughters':[None],
            'new_pole': [new_pole],
            'old_pole': [old_pole]
            }
        self.cells.append(new_cell)
        
        return new_cell['id']
    
    def updatecell(
            self, cell, frame, daughter=None, new_pole=None, old_pole=None
            ):
        
        self.cells[cell]['frames']+=[frame] # Update frame number list
        self.cells[cell]['daughters']+=[daughter]
        self.cells[cell]['old_pole']+=[old_pole]
        self.cells[cell]['new_pole']+=[new_pole]
    
    def setvalue(self,cell,feature,frame,value):
        
        try:
            i = self.cells[cell]['frames'].index(frame)
        except ValueError:
            raise ValueError('Cell %d is not present in frame %f'%(cell,frame))
        
        # Get cell dict
        cell = self.cells[cell]
        
        # If feature doesn't exist yet, create list:
        if feature not in cell:
            cell[feature]=[None for _ in range(len(cell['frames']))]
        
        # Add value:
        cell[feature][i]=value
    
    def getvalue(self,cell,feature,frame):
        
        try:
            i = self.cells[cell]['frames'].index(frame)
        except ValueError:
            raise ValueError('Cell %d is not present in frame %f'%(cell,frame))
        
        return self.cells[cell][feature][i]

class xpreader:
    
    def __init__(
            self, 
            filename,
            channelnames=None,
            use_bioformats=False,
            prototype=None,
            fileorder='pct',
            filenamesindexing=1,
            watchfiles = False
            ):
        '''
        Initialize experiment reader

        Parameters
        ----------
        filename : String
            Path to experiment tif file or directory.If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If no prototype is provided, the filenames 
            are expected to be of the following C-style format: 
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded 
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
                Pos01_Cha3_Fra0005.tif
                p3c1t034.tif
                xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager 
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same 
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing = int
            Selects between 0-based or 1-based indexing in the filename. If 
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1
            

        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the 
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        '''
        
        _, file_extension = os.path.splitext(filename)
        self.filename = filename
        self.use_bioformats = use_bioformats
        self.channelnames = channelnames
        self.watchfiles = watchfiles
        if use_bioformats:
            import bioformats, javabridge
            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(filename)
            md = bioformats.OMEXML(bioformats.get_omexml_metadata(path=filename))
            self.positions = md.get_image_count()
            self.timepoints = md.image(0).Pixels.SizeT # Here I'm going to assume all series have the same number of timepoints
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=False,c=0).dtype
        
        elif os.path.isdir(filename): # Experiment is stored as individual image TIFF files in a folder
            self.filetype = 'dir'
            self.filehandle = filename
            self.fileorder = fileorder
            self.filenamesindexing = filenamesindexing
            # If filename prototype is not provided, guess it from the first file:
            if prototype is None:
                imgfiles = [x for x in os.listdir(filename) if os.path.splitext(x)[1].lower() in ('.tif','.tiff')]
                # Here we assume all images in the folder follow the same naming convention:
                numstrs = re.findall("\d+", imgfiles[0]) # Get digits sequences in first filename
                charstrs = re.findall("\D+", imgfiles[0]) # Get character sequences in first filename
                if len(numstrs)!=3 or len(charstrs)!=4:
                    raise ValueError('Filename formatting error. See documentation for image sequence formatting')
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype =    charstrs[0]+'%0'+str(len(numstrs[0]))+'d'+\
                                    charstrs[1]+'%0'+str(len(numstrs[1]))+'d'+\
                                    charstrs[2]+'%0'+str(len(numstrs[2]))+'d'+\
                                    charstrs[3]
            else:
                self.prototype = prototype
            # Get experiment settings by testing if relevant files exist: 
            # Get number of positions:
            if 'p' in self.fileorder:
                self.positions = 0
                while(os.path.exists(self.getfilenamefromprototype(self.positions,0,0))): self.positions+=1
            else:
                self.positions = 1
            # Get number of channels:
            if 'c' in self.fileorder:
                self.channels = 0
                while(os.path.exists(self.getfilenamefromprototype(0,self.channels,0))): self.channels+=1
            else:
                self.channels = 1
            # Get number of frames/timepoints:
            if 't' in self.fileorder:
                self.timepoints = 0
                while(os.path.exists(self.getfilenamefromprototype(0,0,self.timepoints))): self.timepoints+=1
            else:
                self.timepoints = 1 # I guess this shouldn't really happen
            # Get image specs:
            if self.watchfiles:
                # Start file watcher thread:
                self.watcher = files_watcher(self)
                self.watcher.start()
            else:
                # Load first image, get image data from it
                I = cv2.imread(self.getfilenamefromprototype(0,0,0),cv2.IMREAD_ANYDEPTH)
                self.x = I.shape[1]
                self.y = I.shape[0]
                self.dtype = I.dtype
            
        elif (file_extension.lower() == '.tif' or file_extension.lower() == '.tiff'): # Works with single-series tif & mutli-series ome.tif
            from skimage.external.tifffile import TiffFile
            self.filetype = 'tif'
            self.filehandle = TiffFile(filename)
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[0] # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find('T')]
            self.channels = s.shape[s.axes.find('C')]
            self.x = s.shape[s.axes.find('X')]
            self.y = s.shape[s.axes.find('Y')]
            self.dtype = s.pages[0].asarray().dtype
                
    def close(self):
        if self.use_bioformats:
            self.filehandle.close()
            import javabridge
            javabridge.kill_vm()
        elif self.filetype == 'tif': # Nothing to do if sequence directory
            self.filehandle.close()
            
    def getfilenamefromprototype(self,position,channel,frame):
        '''
        Generate full filename for specific frame based on file path, 
        prototype, fileorder, and filenamesindexing

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        '''
        filenumbers = []
        
        for i in self.fileorder:
            if i=='p':
                filenumbers.append(position+self.filenamesindexing)
            if i=='c':
                filenumbers.append(channel+self.filenamesindexing)
            if i=='t':
                filenumbers.append(frame+self.filenamesindexing)
        return os.path.join(self.filehandle,self.prototype % tuple(filenumbers))
        
        
        
    def getframes(self, positions=None, channels=None, frames=None,
                  squeeze_dimensions=True, 
                  resize=None, 
                  rescale=None,
                  globalrescale=None,
                  rotate=None):
        '''
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer 
            or a tuple/list will be returned. If None is passed, all positions 
            are returned. 
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer 
            or a tuple/list will be returned. If the channel names have been 
            defined, the channel(s) can be passed as a string or tuple/list of 
            strings. If an empty list is passed, None is returned. If None is 
            passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list 
            will be returned. If None is passed, all frames are returned. If -1
            is passed and the file watcher is activated, only new frames are
            read. Works only for one position and channel at a time.
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        globalrescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in all frames to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        '''
        
        # Handle options:
        if positions is None: # No positions specified: load all
            positions = list(range(self.positions))
        elif type(positions) is not list and type(positions) is not tuple:
            positions = [positions]
          
        if channels is None: # No channel specified: load all
            channels = list(range(self.channels))
        elif type(channels) is str: # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels+' is not a valid channel name.')
        elif (type(channels) is list or type(channels) is tuple):
            # If list of ints, nothing to do
            if len(channels)==0: # Empty list of channels: Return None
                return None
            elif all([type(c) is str for c in channels]): # refer to channels by list/tuple of names:
                if self.channelnames is None:
                    raise ValueError('Set channel names first')
                for i, c in enumerate(channels):
                    if c in self.channelnames:
                        channels[i] = self.channelnames.index(c)
                    else:
                        raise ValueError(c+' is not a valid channel name.')   
        elif type(channels) is not list and type(channels) is not tuple:
            channels = [channels]
        
        if frames is None: # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            if frames == -1: # Read new frames from files watcher:
                if self.watchfiles:
                    if len(positions) > 1 or len(channels) > 1:
                        raise ValueError('Can not load latest frames for more than one position/channel')
                    new = self.watcher.new[positions[0]][channels[0]]
                    frames = list(
                        range(
                            self.watcher.old[positions[0]][channels[0]]+1,
                            new+1
                            )
                        )
                else:
                    frames = list(range(self.timepoints))
            else:
                frames = [frames]
        
        # If files watcher, update the old frames array:
        if self.watchfiles:
            for p in positions:
                for c in channels:
                    self.watcher.old[p][c] = frames[-1]
            
        # Allocate memory:
        if rescale is None:
            dt = self.dtype
        else:
            dt = np.float32
        if resize is None:
            output = np.empty([len(positions),len(frames),len(channels),self.y,self.x],dtype=dt)
        else:
            output = np.empty([len(positions),len(frames),len(channels),resize[0],resize[1]],dtype=dt)
        
        # Load images:
        for p, pos in enumerate(positions):
            for c, cha in enumerate(channels):
                for f, fra in enumerate(frames):
                    # Read frame:
                    if self.use_bioformats:
                        frame = self.filehandle.read(series=pos,c=cha,t=fra,rescale=False)
                    elif self.filetype == 'dir':
                        frame = cv2.imread(self.getfilenamefromprototype(pos, cha, fra),cv2.IMREAD_ANYDEPTH)
                    elif self.filetype == 'tif':
                        frame = self.filehandle.series[pos].\
                                                    pages[fra*self.channels+cha].\
                                                    asarray()
                    # Optionally resize and rescale:
                    if rotate is not None:
                        frame = imrotate(frame, rotate)
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1]) #cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame,rescale)
                    # Add to output array:
                    output[p,f,c,:,:] = frame
                    
        # Rescale all images:
        if globalrescale is not None:
            output = rangescale(output,globalrescale)

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output
    


class files_watcher(Thread):
    
    def __init__(self, reader):
        super().__init__()
        self.daemon = True
        self.reader = reader
        self.old = []
        self.new = []
    
    def run(self):
        while(True):
            
            # Run through monitored positions:
            for p in range(self.reader.positions):
                
                if len(self.old) <= p: # Position added / not done yet
                    # Append empty lists to the end:
                    self.old+=[[] for _ in range(p+1-len(self.old))]
                    self.new+=[[] for _ in range(p+1-len(self.new))]
                
                # Run through monitored channels:
                for channel in range(self.reader.channels):
                    
                    if len(self.old[p]) <= channel:
                        # Append empty lists to the end:
                        self.old[p]+=[
                            -1 for _ in range(channel+1-len(self.old[p]))
                            ]
                        self.new[p]+=[-1]*(channel+1-len(self.new[p]))
                    
                    # Check if new files have been written:
                    i = self.new[p][channel]
                    while(
                            os.path.exists(
                                self.reader.getfilenamefromprototype(
                                    p,
                                    channel,
                                    i+1
                                    )
                                )
                            ): i+=1
                    
                    # Store new timepoint number:
                    self.new[p][channel] = i
            time.sleep(.01)
            
    def newfiles(self):
        
        newfiles = []
        for p, pos_old in enumerate(self.old):
            for c, latest_read in enumerate(pos_old):
                if latest_read < self.new[p][c]:
                    newfiles += [(p,c)]
        return newfiles