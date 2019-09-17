from utils.wavdatagenerator import DataGeneratorPatch
from utils.wavutils import load_tensor

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.layers.convolutional import SeparableConv2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras import backend as K
from keras import layers
from keras import models
import tensorflow as tf
import argparse
import os

#ENTRY DATA

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to the feature directory we'll be working on ",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\0.features' )
args = vars(ap.parse_args())
feature_dir = args['path']


test_patch_gen = DataGeneratorPatch(
            feature_dir =feature_dir,
            batch_size = 16,
            input_dim = (51,96),
            n_classes = 1
 )

val_patch_gen = DataGeneratorPatch(
            feature_dir =feature_dir,
            batch_size = 16,
            input_dim = (51,96),
            n_classes = 1,
            type='val'
 )

predict_patch_gen = DataGeneratorPatch(
            feature_dir =feature_dir,
            batch_size = 16,
            input_dim = (51,96),
            n_classes = 1,
            type = 'predict'
 )

def gen_model(shape):
    input = layers.Input(shape=(shape))
    # First conv layer
    c_1 = layers.Conv2D(48,(3,8),padding='same')(input)
    c_2 = layers.Conv2D(32,(3,32),padding='same')(input)
    c_3 = layers.Conv2D(16,(3,64),padding='same')(input)
    c_4 = layers.Conv2D(16,(3,90),padding='same')(input)
    conv_1 = layers.Concatenate()([c_1,c_2,c_3,c_4])
    x = layers.BatchNormalization()(conv_1)
    x = layers.ReLU()(x)
    # x = layers.MaxPooling2D((5,5))(x)
    x = layers.AveragePooling2D((5,5))(x)
    # Second conv layer
    x = layers.Conv2D(224,5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.MaxPooling2D((6,4))(x)
    x = layers.AveragePooling2D((6,4))(x)
    # Output layer
    x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(1,activation='sigmoid')(x)
    model = models.Model(input,x)
    return model

model = gen_model((51,96,1))
# opt = Adam(0.001)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.summary()
model.compile(optimizer=opt,loss ='binary_crossentropy' , metrics=['accuracy'])
epochs=200
hist = model.fit_generator(test_patch_gen,
                            validation_data=val_patch_gen,
                            epochs=epochs,
                            )


history_dict = hist.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, epochs+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Trainingloss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#TEST ON DIRECTORY  :
res = model.predict_generator(predict_patch_gen)
print(res)

#OUTPUT
import pandas as pd
import os
filenames = predict_patch_gen.list_IDs[0:res.shape[0]]
# df = pd.DataFrame(list(zip(filenames,res[:,0],res[:,1])),
#                     columns=['filename','fish_activity','click_activity'])
df = pd.DataFrame(list(zip(filenames,[round(k) for k in res[:,0]])),
                    columns=['filename','fish_activity'])
df.to_csv(os.path.join(feature_dir,'output_labels.csv'))

#Afterward comparison
fish_tab = df['fish_activity']
true_fish_df = pd.read_csv(os.path.join(feature_dir,'labels.csv'))
true_fish_tab = true_fish_df['fish_activity']
# fish_tab = np.asarray([np.float32(k) for k in fish_tab])
print('Accuracy calculated : ')
print(np.mean(np.equal(fish_tab,true_fish_tab)))
