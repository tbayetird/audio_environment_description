from utils import wavvisualisation as wv
from utils import parseutils as pu
import os

path = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\0.features\\01000000_0012_mel.data'
dir = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\0.features\\'
labels = pu.custom_csv_to_dic(os.path.join(dir,'labels.csv'))
for root, dirs, files in os.walk(dir):
    for file in files:
        file_path = os.path.join(dir,file)
        filename = file.replace('_mel.data','')
        label = labels[filename]
        wv.visualize_one_tensor(file_path,file,'Label pour ce log-mel spectrum : {}'.format(label))
