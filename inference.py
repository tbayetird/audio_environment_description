from datetime import datetime
startTime = datetime.now()

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
from keras.models import load_model
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
ap.add_argument("-m", "--model", required=False,
    help="name of the model we will be using. should be in the 'models' directory. default is 'model'  ",
    default ='model' )
args = vars(ap.parse_args())
feature_dir = args['path']

predict_patch_gen = DataGeneratorPatch(
            feature_dir =feature_dir,
            batch_size = 16,
            input_dim = (51,96),
            n_classes = 1,
            type = 'predict'
 )

### LOAD MODEL
model_name = args['model']+'.h5'
dir = os.path.dirname(os.path.abspath(__file__))
dir=os.path.join(dir,'models')
model = load_model(os.path.join(dir,model_name))

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

print(datetime.now() - startTime)
