from utils.wavdataset import  customGenerateFeatureData, prepareLabels
import argparse
import os

#ENTRY DATA
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to the directory where we need to prepare data. Should be the one that contains all the folders for individuals recordings ",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset' )
ap.add_argument("-ty", "--type", required=False,
	help="type of operation thata the dataset will be submitted to - default is 'train', other option is 'test' ",
    default ='train' )
ap.add_argument("-n", "--name", required=False,
	help="name of the directory that will be created to store the features. Should not already exists ",
    default ='0.features' )
ap.add_argument("-t", "--train", required=False,
	help="fraction of the features to use as training data. 0<=train<=1 . Default is 0.8",
    default =0.8 )
ap.add_argument("-v", "--val", required=False,
	help="fraction of the features to use as validation data. 0<=val<=1 . Default is 0.2",
    default =0.2 )
ap.add_argument("-f", "--freq", required=False,
	help="frequency of sampling for the audio samples",
    default =44100)
args = vars(ap.parse_args())



customGenerateFeatureData(args['path'],
                          args['name'],
						  int(args['freq'])
                        )
prepareLabels(args['path'],
			  args['type'],
              args['train'],
              args['val'])
