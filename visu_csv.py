from datetime import datetime
startTime = datetime.now()

from utils import wavvisualisation as wv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to the csv containing the datas we want to visualize ",
    default ='D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\0.features\\labels.csv' )
args = vars(ap.parse_args())
feature_dir = args['path']


wv.visualize_all_from_labels(args['path'])

print(datetime.now() - startTime)
