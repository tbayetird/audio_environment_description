import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import pylab
import wave
import cv2
import os

def decompose_and_save_wav(directorypath,filename,subnframes):
    wav = wave.open(os.path.join(directorypath,filename+'.wav'),'r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    sub_tmp_frames=0
    tmp_frames = []
    while sub_tmp_frames<nframes:
        tmp_frames.append(wav.readframes(subnframes))
        sub_tmp_frames+=subnframes
    wav.close()
    tmp_frames.pop() #We retrieve the last one, because it's not complete (in many cases)
    try :
        os.mkdir(os.path.join(directorypath,filename))
        for i,elem in enumerate(tmp_frames):
            tmp_wav=wave.open(os.path.join(directorypath,filename,filename+'_{}.wav'.format(str(i).zfill(4))),'w')
            tmp_wav.setparams([nchannels,sampwidth,framerate,subnframes,comptype,compname])
            tmp_wav.writeframes(elem)
            tmp_wav.close()
    except FileExistsError :
        pass
    return nframes
# decompose_and_save_wav('D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\','01000000',44100)

def check_activity(lowind,supind,tab,tabind):
    if tab[0][tabind]>=supind:
        return(tabind,0)
    if tab[1][tabind]<=lowind:
        return (tabind,0)
    while(tab[1][tabind]<supind and tabind < (tab[1]).shape[0]-1):
        tabind +=1
    return (tabind,1)

def parse_segmentations_and_write_csv(inputdirectorypath,outputdirectorypath,filename,subnframes,nframes):
    click_csv = pd.read_csv(os.path.join(inputdirectorypath,filename+'_segment_click.csv'))
    fish_csv = pd.read_csv(os.path.join(inputdirectorypath,filename+'_segment_fish.csv'))
    try :
        click_tab = [np.array(click_csv['Segment_click_1']),np.array(click_csv['Segment_click_2'])]
        fish_tab = [np.array(fish_csv['Segment_fish_1']),np.array(fish_csv['Segment_fish_2'])]
    except KeyError:
        click_tab = [[0],[0]]
        fish_tab = [[0],[0]]
    indframe = 0
    indclick = 0
    indfish = 0
    k=0
    full_label_data=[[],[],[]]
    while indframe < (nframes-subnframes): #don't use the last one
        indframesup = indframe + subnframes
        (indclick,clicklabel)=check_activity(indframe,indframesup,click_tab,indclick)
        (indfish,fishlabel)=check_activity(indframe,indframesup,fish_tab,indfish)
        tmp_label_data = {'fish_activity' : [fishlabel],
                            'click_activity' : [clicklabel]}
        tmp_df = pd.DataFrame(tmp_label_data, columns=['fish_activity','click_activity'])
        tmp_df.to_csv(os.path.join(outputdirectorypath,filename+'_{}.csv'.format(str(k).zfill(4))))
        indframe = indframesup
    # Also reconstruct a unique file containing names of the subsample and associated labels
        full_label_data[0].append((filename+'_{}'.format(str(k).zfill(4))))
        full_label_data[1].append(fishlabel)
        full_label_data[2].append(clicklabel)
        k+=1
    full_label_df = pd.DataFrame({'filename' : full_label_data[0],
                                  'fish_activity' : full_label_data[1],
                                  'click_activity' : full_label_data[2]
                                  })
    full_label_df.to_csv(os.path.join(outputdirectorypath,'labels.csv'))

# parse_and_write_csv('D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\SEGMENTS','D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000','01000000',44100,5291776)

def sep_filename_and_ext(file):
    ## Will fail if there is a '.' in the filename/pathname
    file_sep= file.split('.')
    ext = file_sep[len(file_sep)-1]
    filename = file_sep[:len(file_sep)-1][0]
    return(filename,ext)

def prepare_dataset(directory,segmentdirectory,subnframes):
    ##TODO : decompose and save all wav file in a directory and then parse and write CSV for each
    try :
        os.mkdir(os.path.join(directory,'prepared_dataset'))
    except FileExistsError:
        pass
    for root,dirs,files in os.walk(directory):
        for file in files:
            filename,ext = sep_filename_and_ext(file)
            if(ext !='wav'):
                print('file {} passed'.format(file))
                continue
            nframes =decompose_and_save_wav(root,filename,subnframes)
            parse_segmentations_and_write_csv(segmentdirectory,os.path.join(directory,filename),filename,subnframes,nframes)
            os.system('move {} {}'.format(os.path.join(directory,filename),os.path.join(directory,'prepared_dataset')))
        break
    print('Finished preparing dataset')

# prepare_dataset('D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\','D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\SEGMENTS',44100)

from .wavfeatures import genFeatures

def generateFeaturesData(sourceDir,outDir):
    for root,dirs,files in os.walk(sourceDir):
        audio_files_list = [f for f in files if f.endswith('.wav')]
        break
    for f in audio_files_list:
        genFeatures(os.path.join(sourceDir,f),outDir,f,'_mel')

def customGenerateFeatureData(sourceDir,outDir):
    os.mkdir(os.path.join(sourceDir,outDir))
    for root,dirs,files in os.walk(sourceDir):
        for dir in dirs:
            for subroot,subdirs,subfiles in os.walk(os.path.join(root,dir)):
                audio_files_list=[f for f in subfiles if f.endswith('.wav')]
                break
            for f in audio_files_list:
                genFeatures(os.path.join(sourceDir,dir,f),os.path.join(sourceDir,outDir),f,'_mel')
        break

def labelsFusion(pd1,pd2):
        return pd1.append(pd2,ignore_index=True)

# local_dir = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\'
# print(labelsFusion(os.path.join(local_dir,'01000000','labels.csv'),
#                     os.path.join(local_dir,'01001000','labels.csv')))

def fusionAllLabels(dir1):
    first = True
    labels = None
    for root,dirs,files in os.walk(dir1):
        for dir in dirs :
            for root,subdirs,subfiles in os.walk(os.path.join(dir1,dir)):
                for file in subfiles:
                    if file=='labels.csv':
                        csv_path = os.path.join(dir1,dir,file)
                        if first :
                            labels = pd.read_csv(csv_path)
                            first = False
                        else:
                            tmp_label = pd.read_csv(csv_path)
                            labels = labelsFusion(labels,tmp_label  )
                break
        break
    return labels

def prepareLabels(dir,train_frac,val_frac):
    labels = fusionAllLabels(dir)
    labels.to_csv(os.path.join(dir,'0.features','labels.csv'))
    train = labels.sample(frac=train_frac)
    train.to_csv(os.path.join(dir,'0.features','train.csv'))
    val = labels.sample(frac=val_frac)
    val.to_csv(os.path.join(dir,'0.features','val.csv'))
