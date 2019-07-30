from model import unet_seg
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint

# Parameters:
data_folder = 'data/segmentation_set/train/'
target_size = (256, 32)
input_size = target_size + (1,)
batch_size = 10
epochs = 200
steps_per_epoch = 250
model_file = 'unet_moma_seg.hdf5'

#Data generator:
data_gen_args = dict(
                    rotation = 0.5,
                    shiftX=.05,
                    shiftY=.05,
                    zoom=.15,
                    horizontal_flip=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True)

myGene = trainGenerator_seg(batch_size,
                           data_folder + 'img/',
                           data_folder + 'seg/',
                           data_folder + 'wei/',
                           augment_params = data_gen_args,
                           target_size = target_size)


# Define model:
model = unet_seg(input_size = input_size)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)


# Train it:
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])