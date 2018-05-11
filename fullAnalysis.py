# USAGE
# python fullAnalysis.py --conf conf/flowers17.json

# suppress any FutureWarning from Theano
from __future__ import print_function
import warnings

import time


from guppy import hpy

warnings.simplefilter(action="ignore", category=FutureWarning)
# import the necessary packages
from utils.conf import Conf
import argparse

from index_features import generateFeatures
from StatisticalComparison import statisticalComparison
from train import train


# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and grab all image paths in the dataset
conf = Conf(args["conf"])
verbose = False
start = time.time()
# imagePaths = list(paths.list_images(conf["dataset_path"]))
# generateFeatures(conf,imagePaths, verbose)
# statisticalComparison(conf, verbose)
generateFeatures(conf["output_path"], conf["batch_size"], conf["dataset_path"], conf["feature_extractors"], verbose)

statisticalComparison(conf["output_path"], conf["dataset_path"], conf["feature_extractors"], conf["model_classifiers"], conf["measure"], verbose)

end = time.time()
train(conf["output_path"], conf["dataset_path"], conf["training_size"])

print("It has taken " + str(end - start) + " seg to run" )