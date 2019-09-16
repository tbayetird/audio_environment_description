import numpy as np
import os
import utils
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence, to_categorical

from .wavutils import load_tensor
from .parseutils import custom_csv_to_dic, custom_load_list_IDs

class DataGeneratorPatch(Sequence):

    def __init__(self,feature_dir=None,batch_size=64,dtype=np.float32,
                input_dim=(32,32,32),n_classes=1,ltype = np.int32,n_channels = 1,
                list_IDs=[], labels = {},type='train'):
        self.data_dir = feature_dir
        self.dtype = dtype
        self.batch_size= batch_size
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.ltype = ltype
        self.list_IDs= list_IDs
        self.type=type
        if (self.list_IDs==[]):
            self.load_list_IDs()
        self.indexes = np.arange(len(self.list_IDs))
        self.labels = labels
        if (self.labels=={}):
            self.load_default_labels()
        self.n_channels=n_channels
        if type=='train':
            self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        # print('__len__ called and returned {} '.format(int(np.floor(len(self.list_IDs) / self.batch_size))))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self,index):
        'Generate one batch of data'
        # print('__getitem__ called ')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print("\nlist of temporaries IDs : {}".format(list_IDs_temp))
        X,y = self.__data_generation(list_IDs_temp)
        return X,y

    def __data_generation(self, list_IDs_temp):
        'Loads the data for the batch '
        # print('__data_generation called ')
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels) , dtype=self.dtype)
        y = np.empty((self.batch_size, self.n_classes), dtype=self.ltype)

        for i, ID in enumerate(list_IDs_temp):
            try:
                tensor = self.load_ID(ID)
                if (tensor is None):
                    print('this might raise errors')
                    return X,y
                tensor = tensor.reshape(*self.input_dim,self.n_channels)
                X[i,] = tensor
                y[i] = np.array(self.load_label(ID))
            except ValueError :
                print('\n Warning : some of the data seems to be corrupted or of incorrect dimension')
                print('Related tensor : {}'.format(ID))
                print('This could break your training as loss values may be weird. ')
                X[i,]=None
        # print("\n expected labels for this batch : {}".format(y))
        return X,y

    def __get_indexes(self):
        return self.indexes

    def on_epoch_end(self):
        'executes itself at each epoch end '
        self.indexes = np.random.permutation(self.indexes)

    def load_ID(self,ID):
        'Load tensor designated by ID '
        # TODO: generalize extensions
        # print("tensor ID : {}".format(ID))
        tensor = load_tensor(os.path.join(self.data_dir,ID+'_mel.data'),self.input_dim)
        return tensor

    def load_label(self,ID):
        'returns the label designated by ID '
        # print(self.labels)
        return self.labels[ID]

    def load_list_IDs(self):
        'load the list of IDs from data_dir based on extension '
        ##TODO : generalize to any extension or suffixe
        # for root, dirs, files in os.walk(self.data_dir):
        #     features_IDs = [f.replace('_mel.data','') for f in files if f.endswith('.data')]
        #     break
        if self.type=='train':
            csv_path=os.path.join(self.data_dir,'train.csv')
        if self.type=='val':
            csv_path=os.path.join(self.data_dir,'val.csv')
        if self.type=='predict':
            csv_path=os.path.join(self.data_dir,'labels.csv')
        self.list_IDs = custom_load_list_IDs(csv_path)

    def load_default_labels(self):
        'load labels by default if the csv exists '
        if self.type=='train':
            csv_path=os.path.join(self.data_dir,'train.csv')
        if self.type=='val':
            csv_path=os.path.join(self.data_dir,'val.csv')
        if self.type=='predict':
            csv_path=os.path.join(self.data_dir,'labels.csv')
        # print('__load_default_labels called : path is {}'.format(csv_path))
        assert os.path.isfile(csv_path)
        self.labels = custom_csv_to_dic(csv_path)
