from model import unet_track
from data import trainGenerator_track, estimateClassweights
from tensorflow.keras.callbacks import ModelCheckpoint

# Parameters:
data_folder = 'data/tracking_set/train/'
target_size = (256, 32)
input_size = target_size + (4,)
batch_size = 5
epochs = 400
steps_per_epoch = 250
model_file = 'unet_moma_track.hdf5'

#Data generator:
data_gen_args = dict(
                    rotation = 1,
                    shiftX=.05,
                    shiftY=.05,
                    zoom=.15,
                    horizontal_flip=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True)

myGene = trainGenerator_track(batch_size,
                             data_folder + 'img/',
                             data_folder + 'seg/',
                             data_folder + 'previmg/',
                             data_folder + 'segall/',
                             data_folder + 'mother/',
                             data_folder + 'daughter/',
                             data_gen_args,
                             target_size = target_size)

#class_weights = estimateClassweights(myGene) # Not used for mother machine
class_weights = (1,1,1)

# Define model & train it:
model = unet_track(input_size = input_size, class_weights = class_weights)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])