from __future__ import print_function
#import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import tensorflow as tf
import os, glob, re, random, warnings, copy, importlib
import skimage.io as io
import skimage.util as util
import skimage.transform as trans
from skimage.measure import label
from skimage.morphology import square, binary_opening, medial_axis
from scipy import interpolate

# Try to import elastic deformations, issue warning if not found:
if importlib.util.find_spec("elasticdeform") is None:
    warnings.warn("Could not load elastic deformations module.")
else:
    import elasticdeform
        



# UTILITIES:

def binarizerange(i):
    newi = i
    newi[i > (np.amin(i)+np.amax(i))/2] = 1
    newi[i <= (np.amin(i)+np.amax(i))/2] = 0
    return newi

def readreshape(filename, target_size = (256,32), binarize = False, order = 1):
    i = io.imread(filename,as_gray = True)
    i = trans.resize(i, target_size, anti_aliasing=True, order=order)
    if binarize:
        i = binarizerange(i)
    i = np.reshape(i,i.shape + (1,) )
    return i

def data_augmentation(images_input, aug_par, order=0):
    
    #processing inputs / initializing variables::
    output = list(images_input)
    if np.isscalar(order):
        orderlist = [order] * len(images_input)
    else:
        orderlist = list(order)
    
    # Apply augmentation operations:
    
    if "illumination_voodoo" in aug_par:
        if aug_par["illumination_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = illumination_voodoo(item)
    
    if "histogram_voodoo" in aug_par:
        if aug_par["histogram_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = histogram_voodoo(item)
                    
    if "elastic_deformation" in aug_par:
        output = elasticdeform.deform_random_grid(output,
                                                  sigma=aug_par["elastic_deformation"]["sigma"],
                                                  points=aug_par["elastic_deformation"]["points"],
                                                  order=[i*3 for i in orderlist], # Using bicubic interpolation instead of bilinear here
                                                  mode='nearest',
                                                  axis=(0,1),
                                                  prefilter=False)
    
    if "horizontal_flip" in aug_par:
        if aug_par["horizontal_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.fliplr(item)
                    
    if "vertical_flip" in aug_par:
        if aug_par["vertical_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.flipud(item)
                    
    if "rotations_90d" in aug_par: # Only works with square images right now!
        if aug_par["rotations_90d"]:
            rot = random.randint(0,3)*90
            if rot>0: 
                for index, item in enumerate(output):
                    output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    if "rotation" in aug_par:
        rot = random.uniform(-aug_par["rotation"],aug_par["rotation"])
        for index, item in enumerate(output):
            output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    if "zoom" in aug_par:
        zoom = random.expovariate(3*1/aug_par["zoom"]) # I want most of them to not be too zoomed
        zoom = aug_par["zoom"] if zoom > aug_par["zoom"] else zoom
    else:
        zoom = 0
        
    if "shiftX" in aug_par:
        shiftX = random.uniform(-aug_par["shiftX"],aug_par["shiftX"])
    else:
        shiftX = 0
        
    if "shiftY" in aug_par:
        shiftY = random.uniform(-aug_par["shiftY"],aug_par["shiftY"])
    else:
        shiftY = 0
        
    if any(abs(x)>0 for x in [zoom,shiftX,shiftY]):
        for index, item in enumerate(output):
            output[index] = zoomshift(item,zoom+1,shiftX,shiftY, order=orderlist[index])
    
    return output

def zoomshift(I,zoomlevel,shiftX,shiftY, order=0):
    oldshape = I.shape
    I = trans.rescale(I,zoomlevel,mode='edge',multichannel=False, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I,(shiftY, shiftX),order=order) # For some reason it looks like X & Y are inverted?
    i0 = (round(I.shape[0]/2 - oldshape[0]/2), round(I.shape[1]/2 - oldshape[1]/2))
    I = I[i0[0]:(i0[0]+oldshape[0]), i0[1]:(i0[1]+oldshape[1])]
    return I
    
def shift(image, vector, order=0):
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode='edge',order=order)

    return shifted

def histogram_voodoo(image,num_control_points=3):
    '''
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''
    control_points = np.linspace(0,1,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)
    
    return mapping(image)

def illumination_voodoo(image,num_control_points=5):
    '''
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber
    '''
    
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0,image.shape[0]-1,num=num_control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(image,
                           np.reshape(
                                   np.tile(
                                           np.reshape(curve,curve.shape + (1,)), (1, image.shape[1])
                                           )
                                   ,image.shape
                                   )
                           )
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))
    
    return newimage

def postprocess(images,square_size=5):
    ''' 
    In this function I post-process outputs to binarize them and do some light
    mathematical morphology to remove small lone regions
    '''
    selem = square(square_size)
    for index, I in enumerate(images):
        I = binarizerange(I)
        I = binary_opening(I,selem=selem)
        images[index,:,:] = I
    return images
        
def weightmap(mask, classweights, params):
    '''
    This function computes the weight map as described in the original U-Net
    paper to force the model to learn borders 
    (Not used, too slow. We preprocess the weight maps in matlab)
    '''

    lblimg,lblnb = label(mask[:,:,0],connectivity=1,return_num=True)
    distance_array = float('inf')*np.ones(mask.shape[0:2] + (max(lblnb,2),))
    for i in range(0,lblnb):
        [_,distance_array[:,:,i]] = medial_axis((lblimg==i+1)==0,return_distance=True)
    distance_array = np.sort(distance_array,axis=-1)
    weightmap = params["w0"]*np.exp(-np.square(np.sum(distance_array[:,:,0:2],-1))/(2*params["sigma"]^2))
    weightmap = np.multiply(weightmap,mask[:,:,0]==0)
    weightmap = np.add(weightmap,(mask[:,:,0]==0)*classweights[0])
    weightmap = np.add(weightmap,(mask[:,:,0]==1)*classweights[1])
    weightmap = np.reshape(weightmap, weightmap.shape + (1,))
    
    return weightmap

# SEGMENTATION FUNCTIONS:

def trainGenerator_seg(batch_size,
                    img_path, mask_path, weight_path,
                    target_size = (256,32),
                    augment_params = {},
                    preload = False,
                    seed = 1):
    '''
    So I turned the data compiler into a generator. I do the transformations/augmentation
    myself because I wasn't entirely happy with the ImageDataGenerator output
    '''
    preload_mask = []
    preload_img = []
    preload_weight = []
    
    # Get training image files list:
    image_name_arr = glob.glob(os.path.join(img_path,"*.png"))
    
    # If preloading, load the images and compute weight maps:
    if preload:
        for filename in image_name_arr:
            preload_img.append(readreshape(filename, target_size = target_size, order = 1))
            preload_mask.append(readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0))
            preload_weight.append(readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0))
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    while True:
        # Reset image arrays:
        image_arr = []
        mask_arr = []
        weight_arr = []
        for _ in range(batch_size):
            # Pick random image index:
            index = random.randrange(0,len(image_name_arr))
            
            if preload:
                # Get from preloaded arrays:
                img = preload_img[index]
                mask = preload_mask[index]
                weight = preload_weight[index]
            else:
                # Read images:
                filename = image_name_arr[index]
                img = readreshape(filename, target_size = target_size, order = 1)
                mask = readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0)
                weight = readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0)

            # Data augmentation:
            [img, mask, weight] = data_augmentation([img, mask, weight], augment_params, order=[1,0,0])
            
            # Append to output list:
            image_arr.append(img)
            mask_arr.append(mask)
            weight_arr.append(weight)

        # Concatenate masks and weights: (gets unstacked by the loss function)
        image_arr = np.array(image_arr)
        mask_wei_arr = np.concatenate((mask_arr,weight_arr),axis=-1)
        yield (image_arr, mask_wei_arr)


def saveResult_seg(save_path,npyfile, files_list = []):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        if files_list:
            filename = os.path.join(save_path,files_list[i])
        else:
            filename = os.path.join(save_path,"%d_predict.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filename,(img*255).astype(np.uint8))


def predictGenerator_seg(files_path, files_list = [], target_size = (256,32)):
    if not files_list:
        files_list = os.listdir(files_path)
    for index, fname in enumerate(files_list):
        img = readreshape(os.path.join(files_path,fname),target_size=target_size,order=1)
        img = np.reshape(img,(1,)+img.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
        yield img

        
        
        
        
        
        
        
        
        
        
        
        
        
        
# TRACKING FUNCTIONS
    
def trainGenerator_track(batch_size,
                    img_path, seg_path, previmg_path, segall_path,
                    mother_path, daughter_path,
                    augment_params = {},
                    target_size = (256,32),
                    seed = 1):
    
    # Initialize variables and arrays:
    image_name_arr = glob.glob(os.path.join(img_path,"*.png"))
    
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    while True:
        # Reset image arrays:
        img_arr = []
        seg_arr = []
        previmg_arr = []
        segall_arr = []
        mother_arr = []
        daughter_arr = []
        bkgd_arr = []

        for _ in range(batch_size):
            # Pick random image file name:
            index = random.randrange(0,len(image_name_arr))
            filename = image_name_arr[index]
            
            # Read images:
            img = readreshape(filename, target_size = target_size, order = 1)
            seg = readreshape(os.path.join(seg_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0)
            previmg = readreshape(os.path.join(previmg_path,os.path.basename(filename)), target_size = target_size, order = 1)
            segall = readreshape(os.path.join(segall_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0)

            mother = readreshape(os.path.join(mother_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0)
            daughter = readreshape(os.path.join(daughter_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0)
            
            # Create the "background" image: (necessary for categorical cross-entropy)
            bkgd = 1 - (mother + daughter)
            
            # Data augmentation:
            [img, seg, previmg, segall, mother, daughter, bkgd] = data_augmentation([img, seg, previmg, segall, mother, daughter, bkgd],
                                augment_params, order=[1,0,1,0,0,0,0])
            
            # Append to arrays:
            seg_arr.append(seg)
            img_arr.append(img)
            previmg_arr.append(previmg)
            segall_arr.append(segall)
            mother_arr.append(mother)
            daughter_arr.append(daughter)
            bkgd_arr.append(bkgd)
            
        # Concatenare and yield inputs and output:
        inputs_arr = np.concatenate((img_arr,seg_arr,previmg_arr,segall_arr),axis=-1)
        outputs_arr = np.concatenate((mother_arr,daughter_arr,bkgd_arr),axis=-1)
        yield (inputs_arr, outputs_arr)
            
            
def predictGenerator_track(test_path,files_list = [],target_size = (256,32)):
    
    if not files_list:
        files_list = os.listdir(test_path)
    for index, item in enumerate(files_list):
            img = readreshape(os.path.join(test_path + 'img/',os.path.basename(item)), target_size = target_size, order = 1)
            seg = readreshape(os.path.join(test_path + 'seg/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0)
            previmg = readreshape(os.path.join(test_path + 'previmg/',os.path.basename(item)), target_size = target_size, order = 1)
            segall = readreshape(os.path.join(test_path + 'segall/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0)
            
            inputs_arr = np.concatenate((img,seg,previmg,segall),axis=-1)
            inputs_arr = np.reshape(inputs_arr,(1,)+inputs_arr.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            
            yield inputs_arr


def predictCompilefromseg_track(img_path,seg_path, files_list = [],
                                  target_size = (256,32)):
    ''' 
    This function generates input data of the same for as predictGenerator_track()
    But it does it firectly from the segmentation results out of the segmentation
    version of this U-Net
    '''
    
    img_arr = []
    seg_arr = []
    previmg_arr = []
    segall_arr = []
    seg_name_list = []
    
    if not files_list:
        files_list = os.listdir(img_path)
        
    for index, item in enumerate(files_list):
            filename = os.path.basename(item)
            # Get position, chamber & frame numbers:
            (pos, cha, fra) = list(map(int, re.findall("\d+",filename)))
            if fra > 1:
                prevframename = 'Position' + str(pos).zfill(2) + \
                                '_Chamber' + str(cha).zfill(2) + \
                                '_Frame' + str(fra-1).zfill(3) + '.png'
                img = readreshape(os.path.join(img_path,filename), target_size = target_size, order = 1)
                segall = readreshape(os.path.join(seg_path,filename), target_size = target_size, order = 0, binarize = True)
                previmg = readreshape(os.path.join(img_path,prevframename), target_size = target_size, order = 1)
                prevsegall = readreshape(os.path.join(seg_path,prevframename), target_size = target_size, order = 0, binarize = True)
                
                lblimg,lblnb = label(prevsegall[:,:,0],connectivity=1,return_num=True)
                
                for lbl in range(1,lblnb+1):
                    segfilename = 'Position' + str(pos).zfill(2) + \
                                '_Chamber' + str(cha).zfill(2) + \
                                '_Frame' + str(fra).zfill(3) + \
                                '_Cell' + str(lbl).zfill(2) + '.png'

                    seg = lblimg == lbl
                    seg = np.reshape(seg, seg.shape + (1,))
                    seg.astype(segall.dtype) # Output is boolean otherwise

                    seg_arr.append(seg)
                    img_arr.append(img)
                    previmg_arr.append(previmg)
                    segall_arr.append(segall)
                    seg_name_list.append(segfilename)
    
    inputs_arr = np.concatenate((img_arr,seg_arr,previmg_arr,segall_arr),axis=-1)
    return inputs_arr, seg_name_list


def estimateClassweights(gene, num_samples = 30):
    '''
    This function estimates the class weights to use with the weighted 
    categorical cross-entropy based on the output of the trainGenerator_track
    output.
    '''
    
    sample = next(gene)
    class_counts = [0] * sample[1].shape[-1] # List of 0s
    
    # Run through samples and classes/categories:
    for _ in range(num_samples):
        for i in range(sample[1].shape[-1]):
            class_counts[i] += np.mean(sample[1][..., i])
        sample = next(gene)
        
    # Warning! If 0 elements of a certain class are present in the samples, the
    # weight for this class will be set to 0. This is for the tracking case
    # (Where there are only empty daughter images in the training set)
    # Try changing the num_samples value if this is a problem
    class_weights = [(x/num_samples)**-1 if x!= 0 else 0 for x in class_counts] # Normalize by nb of samples and invert to get weigths, unless x == 0 to avoid Infinite weights or errors
    
    return class_weights


def saveResult_track(save_path,npyfile, files_list = None, num_class = 2):
    mothers = npyfile[:,:,:,0]
    daughters = npyfile[:,:,:,1]
    for i,mother in enumerate(mothers):
        if files_list:
            filenameMo = os.path.join(save_path,'mother_' + files_list[i])
        else:
            filenameMo = os.path.join(save_path,"%d_mother.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filenameMo,(mother*255).astype(np.uint8))
    for i,daughter in enumerate(daughters):
        if files_list:
            filenameDa = os.path.join(save_path,'daughter_' + files_list[i])
        else:
            filenameDa = os.path.join(save_path,"%d_daughter.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filenameDa,(daughter*255).astype(np.uint8))


       
# simultaneous SEGMENTATION + TRACKING:
        
def predictGenerator_segtrack(test_path,files_list = [],target_size = (256,32)):
    '''
    This is the same as the predict generator for tracking, except that it is
    feeding the 'seg' images instead of the 'segall' ones. In effect, this 
    removes the current segmentation from the inputs
    '''
    
    if not files_list:
        files_list = os.listdir(test_path)
    for index, item in enumerate(files_list):
            img = readreshape(os.path.join(test_path + 'img/',os.path.basename(item)), target_size = target_size, order = 1)
            seg = readreshape(os.path.join(test_path + 'seg/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0)
            previmg = readreshape(os.path.join(test_path + 'previmg/',os.path.basename(item)), target_size = target_size, order = 1)
            # Hacky:
            segall = readreshape(os.path.join(test_path + 'seg/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0)
            
            inputs_arr = np.concatenate((img,seg,previmg,segall),axis=-1)
            inputs_arr = np.reshape(inputs_arr,(1,)+inputs_arr.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            
            yield inputs_arr