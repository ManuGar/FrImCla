from __future__ import absolute_import
import h5py
import argparse
from .utils.conf import Conf
import pandas as pd
import numpy as np

# Small program to convert an h5 file to a csv

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())
conf = Conf(args["conf"])
featuresPath = conf["features_path"][0:conf["features_path"].rfind(".")] + "-"+ conf["model"] +".hdf5"
db = h5py.File(featuresPath)
names = db["image_ids"]
names = [x.split(":")[1] for x in names] # [x.split(":")[0] + "/" +
columns = [conf["model"] + "-" + str(x) for x in range(0,db["features"].shape[1])]
df1 = pd.DataFrame([x for x in db["features"]],index=names,columns=columns)
featuresCSVPath = conf["features_csv_path"][0:conf["features_csv_path"].rfind(".")] + "-"+ conf["model"] +".csv"
df1.to_csv(featuresCSVPath, encoding = 'utf8')

# En caso de ser muy pesado el fichero utilizar esta opcion
# i=0
# features = db["features"]
# while i < len(features):
#     print("Iteration " + str(i%100+1))
#     df1 = pd.DataFrame([x for x in features[i:i+100]], index=names[i:i+100], columns=columns)
#     featuresCSVPath = conf["features_csv_path"][0:conf["features_csv_path"].rfind(".")] + "-" + str(i) +"-" + conf["model"] + ".csv"
#     df1.to_csv(featuresCSVPath, encoding='utf8')
#     i=i+100
