from data import saveResult_seg, predictGenerator_seg, postprocess
from model import unet_seg
from os import listdir

# Files:
inputs_folder = 'C:\\DeepLearning\\data\\segmentation\\img\\'
save_folder = 'C:\\DeepLearning\\data\\segmentation\\output\\'
model_file= 'unet_moma_seg.hdf5'
unprocessed = listdir(inputs_folder)

# Parameters:
target_size = (256, 32)
input_size = target_size + (1,)
process_size = 4096

# Load up model:
model = unet_seg(input_size = input_size)
model.load_weights(model_file)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(process_size,len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Predict:
    predGene = predictGenerator_seg(inputs_folder, files_list = to_process, target_size = target_size)
    results = model.predict_generator(predGene,len(to_process),verbose=1)
    
    # Post process results:
    results[:,:,:,0] = postprocess(results[:,:,:,0])
    
    # Save to disk:
    saveResult_seg(save_folder,results, files_list = to_process)