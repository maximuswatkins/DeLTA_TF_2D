from data import saveResult_seg, predictGenerator_seg, saveResult_track, predictCompilefromseg_track, postprocess
from model import unet_seg, unet_track
from os import listdir
import time

DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
input_folder = 'E:/JBL/onlinexp/delta/img'
seg_folder = 'E:/JBL/onlinexp/delta/seg_output'
track_folder = 'E:/JBL/onlinexp/delta/track_output'

model_file_seg = DeLTA_data + 'mother_machine/models/unet_moma_seg.hdf5'
model_file_track = DeLTA_data + 'mother_machine/models/unet_moma_track.hdf5'

# Get delta directory file list (if not starting from empty:
before = dict ([(f, None) for f in listdir(input_folder)])


# Parameters:
target_size = (256, 32)
input_size = target_size + (1,)

# Load up segmentation model:
model_seg = unet_seg(input_size = input_size)
model_seg.load_weights(model_file_seg)

# Load up tracking model:
model_track = unet_track(input_size = input_size)
model_track.load_weights(model_file_track)

while 1:
    time.sleep(1)
    
    #monitor folder
    after = dict ([(f, None) for f in listdir(input_folder)])
    to_process = [f for f in after if not f in before]
    before = after
    
    if to_process:
        
        # segment new images:
        predGene = predictGenerator_seg(input_folder, files_list = to_process, target_size = target_size)
        results = model_seg.predict_generator(predGene[0],len(to_process),verbose=1)
        results[:,:,:,0] = postprocess(results[:,:,:,0])
        
        # write segmentation output:
        saveResult_seg(seg_folder,results, files_list = to_process)
        
        # track:
        inputs, seg_filenames = predictCompilefromseg_track(input_folder, seg_folder,
                                                            files_list = to_process,
                                                            target_size = target_size)
        if inputs.size:
            # Predict:
            results = model_track.predict(inputs,verbose=1)
            
            # write tracking output:
            saveResult_track(track_folder,results, files_list = seg_filenames)