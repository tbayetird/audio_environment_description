from utils.wavdataset import prepare_dataset
import argparse
import os

#ENTRY DATA
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to the dataset that needs to be prepared",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\' )
ap.add_argument("-t", "--type", required=False,
	help="type of treatment : 'train' will also prepare labels, 'test' will only set up the dataset",
    default ='train' )
ap.add_argument("-s", "--segments", required=False,
	help="path to the segmentation data that will be used for labelling our data - should only be used if type is train ",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\SEGMENTS' )
ap.add_argument("-f", "--freq", required=False,
	help="frequency of sampling for the audio samples",
    default =44100)
args = vars(ap.parse_args())

prepare_dataset(args['path'],
				args['type'],
                args['segments'],
                int(args['freq']))
