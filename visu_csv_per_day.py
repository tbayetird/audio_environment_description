from datetime import datetime
startTime = datetime.now()

from utils import wavvisualisation as wv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to the csv containing the datas we want to visualize ",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\0.features\\output_labels.csv' )
ap.add_argument("-f", "--format", required=False,type=int,
	help="format of the data : 2018 or 2019, default is 2019 ",
    default =2019 )
args = vars(ap.parse_args())
feature_dir = args['path']


wv.visualize_all_from_labels_per_day(args['path'],
                                     format=int(args['format']))

print(datetime.now() - startTime)
