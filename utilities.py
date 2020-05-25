import cv2, os
import numpy as np
from skimage.external.tifffile import TiffFile

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
    return img[box['ytl']:box['ybr'],box['xtl']:box['xbr']]

def rangescale(frame, rescale):
    '''
    Rescale image values to be within range

    Parameters
    ----------
    frame : 2D numpy array
        Input image.
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    '''
    frame = frame.astype(np.float32)
    frame = ((frame-np.min(frame))/np.ptp(frame))*np.ptp(rescale)+rescale[0]
    return frame

def driftcorr(img,template=None,box=None, drift=None):
    '''
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array
        The frames to correct drfit for.
    template : None or 2D numpy array, optional
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
    2D numpy array, tuple of len 2
        Drift-corrected image and drift.

    '''
    
    if len(img.shape)==2:
        twoDflag=True
        np.expand_dims(img,axis=0)
    else:
        twoDflag=False
        
    if drift is None:
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


def getChamberBoxes(chambersmask):
    '''
    This function extracts the bounding boxes of the chambers in thebinary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D numpy array
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of dictionaries
        List of cropping box dictionaries (see cropbox()).

    '''
    chamberboxes = []
    chamberimg = cv2.threshold(chambersmask,.5,1,cv2.THRESH_BINARY)[1].astype('uint8')
    # chamberimg = cv2.resize(chamberimg,scope_image_size, interpolation=cv2.INTER_NEAREST)
    contours = cv2.findContours(chamberimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for chamber in contours[1]:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamberboxes.append(dict(
                xtl = xtl,
                ytl = ytl-25, # Added -25 to make sure the top isn't cropped
                xbr = xtl+boxwidth,
                ybr = ytl+boxheight)) # tl = top left, br = bottom right.
    chamberboxes.sort(key=lambda elem: elem['xtl'])# Sorting by top-left X (normally sorted by Y top left)
    return chamberboxes


def getDriftTemplate(chamberboxes,img):
    '''
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getChamberBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.

    Returns
    -------
    2D numpy array
        A cropped region of the image to use as template for drift correction
        (see driftcorr()).

    '''
    (y_cut,x_cut) = [round(i*.025) for i in img.shape] # Cutting out 2.5% of the image on eahc side as drift margin
    box = dict(
            xtl = x_cut,
            xbr = -x_cut,
            ytl = y_cut,
            ybr = max(chamberboxes,key=lambda elem: elem['ytl'])['ytl']-y_cut)
    return cropbox(img,box)

def getSinglecells(seg):
    seg = cv2.threshold(seg,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
    singlecellseg = np.empty([len(contours),256,32])
    for c, contour in enumerate(contours):
        singlecellseg[c] = cv2.drawContours(np.zeros((256,32), dtype=np.uint8),[contour],0, color=1, thickness=-1)
    return singlecellseg

def getTrackingScores(inputs,outputs):
    targetcells = getSinglecells(inputs[0,:,:,3]).astype(np.uint8)
    scores = np.zeros([outputs.shape[0],targetcells.shape[0],2],dtype=np.float32)
    for o in range(outputs.shape[0]):
        mother_pred = cv2.threshold(outputs[o,:,:,0],.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
        daughter_pred = cv2.threshold(outputs[o,:,:,1],.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
        for i in range(targetcells.shape[0]):
            scores[o,i,0] = getOverlap(mother_pred,targetcells[i,:,:])
            scores[o,i,1] = getOverlap(daughter_pred,targetcells[i,:,:])
    return scores
        
def getOverlap(output,target):
    return  cv2.sumElems(
                    cv2.bitwise_and(output,
                                    target)
                )[0]/\
            cv2.sumElems(
                    target
                )[0]

def getAttributions(scores):
    attrib = np.zeros(scores.shape)
    for i in range(scores.shape[1]): # Go over cells in the *new* frame
        if np.sum(scores[:,i,:]>0) == 1: # If only one old cell was attributed to the new cell
            attrib[:,i,:] = scores[:,i,:]>0
        elif np.sum(scores[:,i,:]>0) > 1: # If conflicts
            if np.sum(scores==np.max(scores[:,i,:])) == 1: # One cell has a higher score than the others
                attrib[:,i,:] = scores[:,i,:]==np.max(scores[:,i,:])
            elif np.any(scores[:,i,1]==np.max(scores[:,i,:])): # At least one of the highest scoring old cells would be the mother of this new "daughter"
                attrib[np.argmax(scores[:,i,1]),i,1]=True # keep only the first one (which is also the one higher in the image)
            else: # If only mother-to-mother couplings are in conflict
                attrib[np.argmax(scores[:,i,0]),i,0]=True # keep only the first one (which is also the one higher in the image)
    for o in range(scores.shape[0]): # Go over cells in the *old* frame
        if np.sum(attrib[o,:,0]>0) > 1: # If one old cell gets attributed to more than one new cell (mother-mother coupling)
            attrib[o,np.argmax(attrib[o,:,0]),0]=True # keep only the first one
        if np.sum(attrib[o,:,1]>0) > 1: # If one old cell gets attributed to more than one new cell (mother-daughter coupling)
            attrib[o,np.argmax(attrib[o,:,1]),1]=True # keep only the first one
        if np.any(attrib[o,:,0]) and np.any(attrib[o,:,1]) and np.argmax(attrib[o,:,0]) > np.argmax(attrib[o,:,1]): # If somehow the mother is lower than the daughter
            attrib[o,np.argmax(attrib[o,:,0]),0] = False
            attrib[o,np.argmax(attrib[o,:,0]),1] = True
            attrib[o,np.argmax(attrib[o,:,1]),1] = False
            attrib[o,np.argmax(attrib[o,:,1]),0] = True
    return attrib

def label_seg(seg,cellnumbers=None):
    seg = cv2.threshold(seg,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
    label = np.zeros(seg.shape,dtype=np.uint16) # Because we use uint16, we can not have >65535 cells in one movie
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(label,[contour],c+1 if cellnumbers is None else cellnumbers[c])
    return label

def updatelineage(newframe, stack, framenb=0, lineage=None, attrib=None):
    
    if lineage==None: # Not initialized yet
        lineage = []
        stack[framenb] = label_seg(newframe)
        cells = np.unique(stack[framenb])
        for c in cells[1:]: # Avoiding cell==0 (background)
            lineage.append(createcell(framenb))
    else: # Update according to attribution matrix
        # Init cell numbers, retrieve previous stack frame:
        cellnumbers = []
        label_old = label_seg((stack[framenb-1]>0).astype(np.uint8))
        
        # If previous stack frame has no cells:
        if not np.any(label_old>0): # This can happen if the chamber is empty and suddenly cells appear (can be a segmentation error, or a focus problem)
            contours = cv2.findContours(newframe.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] # In this case just label
            for _ in range(len(contours)):
                lineage.append(createcell(framenb))
                cellnumbers.append(len(lineage))
                
        # If previous stack frame has cells:
        else:
            for i in range(attrib.shape[1]):
                if np.any(attrib[:,i,0]): # If cell is tracked from an old cell: (mother-mother)
                    cellnum = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,0])[0]+1)[0]][0]) # retrieving cell number from stack
                    lineage[cellnum-1]['framenbs'].append(float(framenb)) # Update frame number list
                    if len(lineage[cellnum-1]['daughters']) < len(lineage[cellnum-1]['framenbs']): # If framenbs hasn't been updated (in case somehow daughter goes through before mother)
                        lineage[cellnum-1]['daughters'].append(0.)
                    cellnumbers.append(cellnum)
                elif np.any(attrib[:,i,1]): # Cell is tracked as a daughter cell (mother-daughter)
                    mothernb = int(stack[framenb-1][label_old==(np.nonzero(attrib[:,i,1])[0]+1)[0]][0]) # retrieving cell number from stack
                    # Create new cell, as daughter of mother:
                    lineage.append(createcell(framenb,mothernb))
                    cellnumbers.append(len(lineage))
                    # Update daughters of mother cell:
                    if len(lineage[mothernb-1]['daughters']) < len(lineage[mothernb-1]['framenbs']):
                        lineage[mothernb-1]['daughters'].append(float(cellnumbers[-1]))
                    else:
                        lineage[mothernb-1]['daughters'][-1] = float(cellnumbers[-1])
                else: # Orphan cell
                    lineage.append(createcell(framenb))
                    cellnumbers.append(len(lineage))
                    
        # Compile new labeled frame, push to stack:
        stack[framenb] = label_seg(newframe,cellnumbers=cellnumbers)
    return lineage, stack
        

def createcell(framenb, mothernb=0):
    return dict(framenbs=[float(framenb)], # For compatibility with old version of the pipeline
                mothernb=mothernb,
                daughters=[0.])

class xpreader:
    
    def __init__(self, filename, channelnames=None, use_bioformats=False):
        _, file_extension = os.path.splitext(filename)
        self.channelnames = channelnames
        if use_bioformats: # Unfortunately bioformats only works with python 2, leaving it here if it helps anybody.
            import bioformats, javabridge
            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(filename)
            md = bioformats.OMEXML(bioformats.get_omexml_metadata(path=filename))
            self.positions = md.get_image_count()
            self.timepoints = md.image(0).Pixels.SizeT # Here I'm going to assume all series have the same format
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=False,c=0).dtype
            
        elif (file_extension.lower() == '.tif' or file_extension.lower() == '.tiff'): # Works with single-series tif & mutli-series ome.tif
            self.filetype = 'tif'
            self.use_bioformats = use_bioformats
            self.filehandle = TiffFile(filename)
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[0] # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find('T')]
            self.channels = s.shape[s.axes.find('C')]
            self.x = s.shape[s.axes.find('X')]
            self.y = s.shape[s.axes.find('Y')]
            self.dtype = s.pages[0].asarray().dtype
                
        
        
    def getframes(self, frames=None, positions=None, channels=None, squeeze_dimensions=True, resize=None, rescale=None):
        '''
        Get frames from experiment.

        Parameters
        ----------
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list 
            will be returned. If None is passed, all frames are returned. 
            The default is None.
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer 
            or a tuple/list will be returned. If None is passed, all positions 
            are returned. 
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer 
            or a tuple/list will be returned. If the channel names have been 
            defined, the channel(s) can be passed as a string or tuple/list of 
            strings. If None is passed, all channels are returned.
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

        Raises
        ------
        ValueError
            For various different input errors.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        '''
        
        # Handle options:
        if frames is None: # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            frames = [frames]
            
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
        elif (type(channels) is list or type(channels) is tuple) and all([type(c) is str for c in channels]): # refer to channels by list/tuple of names:
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            for i, c in enumerate(channels):
                if c in self.channelnames:
                    channels[i] = self.channelnames.index(c)
                else:
                    raise ValueError(c+' is not a valid channel name.')   
        elif type(channels) is not list and type(channels) is not tuple:
            channels = [channels]
            
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
                    elif self.filetype == 'tif':
                        frame = self.filehandle.series[pos].\
                                                    pages[fra*self.channels+cha].\
                                                    asarray()
                    # Optionally resize and rescale:
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1]) #cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame,rescale)
                    # Add to output array:
                    output[p,f,c,:,:] = frame
        
        # Return:
        return np.squeeze(output) if squeeze_dimensions else output