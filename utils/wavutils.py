import os
import scipy
import librosa
import soundfile
import numpy as np

###############################
# Some of these functions have been inspired by the github edufonseca/icassp19
# https://github.com/edufonseca/icassp19
###############################

def load_audio_file(file_path):
    data, fs = soundfile.read(file=file_path)
    data = data.T

    #Normalize data
    mean_value = np.mean(data)
    data -= mean_value
    max_value = max(abs(data)) + 0.05 #avoid per zero div
    data = data/max_value
    data = np.reshape(data,[-1,1])
    return data

def save_tensor(var,outPath,name,suffix):
    assert os.path.isdir(outPath), "path to save the tensor does not exist : {}".format(outPath)
    var.tofile(os.path.join(outPath,name).replace('.wav',suffix + '.data'))

def load_tensor(tensor_path,shape):
    ' Load a binary .data file '
    assert os.path.isdir(os.path.dirname(tensor_path)), " path to load tensor does not exist"
    try :
        tensor = np.fromfile(tensor_path)
        tensor = tensor.reshape(shape)
    except FileNotFoundError:
        print(" tried to load tensor but file does not exist")
        print(" tensor name : {}".format(tensor_path))
        return None
    except ValueError :
        print("\n Probably tried to reshape the tensor in a wrong way ")
        print("tensor name : {}".format(tensor_path))
        print("tensor shape : {}".format(tensor.shape))
        print('This could break your training process as your losses might not react well to None values')
        return None
    return tensor
