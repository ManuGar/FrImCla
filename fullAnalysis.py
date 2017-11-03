# USAGE
# python fullAnalysis.py --conf conf/flowers17.json

# suppress any FutureWarning from Theano
from __future__ import print_function
from mpi4py import MPI
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# import the necessary packages
from utils.conf import Conf
from imutils import paths
import argparse

from index_features import generate_features
from StatisticalComparison import statisticalComparison



# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and grab all image paths in the dataset
conf = Conf(args["conf"])
imagePaths = list(paths.list_images(conf["dataset_path"]))
generate_features(conf,imagePaths)
statisticalComparison(conf)

