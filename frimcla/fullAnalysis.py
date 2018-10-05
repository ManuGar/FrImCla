# USAGE
# python fullAnalysis.py --conf conf/flowers17.json
# import the necessary packages
from __future__ import print_function
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import argparse
import os
from utils.conf import Conf
from index_features import generateFeatures
from StatisticalComparison import statisticalComparison
from train import train

# suppress any FutureWarning from Theano

def fullAnalysis(config):
    conf = Conf(config)
    verbose = False
    start1 = time.time()

    generateFeatures(conf["output_path"], conf["batch_size"], conf["dataset_path"], conf["feature_extractors"], verbose)
    end1 = time.time()

    start2 = time.time()
    statisticalComparison(conf["output_path"], conf["dataset_path"], conf["feature_extractors"],
                          conf["model_classifiers"], conf["measure"], verbose)

    end2 = time.time()
    start3 = time.time()
    train(conf["output_path"], conf["dataset_path"], conf["training_size"])
    end3 = time.time()
    dataset = conf["dataset_path"][conf["dataset_path"].rfind("/"):]

    f = open(os.path.abspath(conf["output_path"]+ dataset + "/timeFile.txt"), "w")
    f.write("It has taken " + str(end1 - start1) + " seg to generate the features\n")
    f.write("It has taken " + str(end2 - start2) + " seg to generate the model comparison and the statistical analysis\n")
    f.write("It has taken " + str(end3 - start3) + " seg to train the best model\n")
    f.write("It has taken " + str(end3 - start1) + " seg to run\n")
    print("It has taken " + str(end1 - start1) + " seg to generate the features")
    print("It has taken " + str(end2 - start2) + " seg to generate the model comparison and the statistical analysis")
    print("It has taken " + str(end3 - start3) + " seg to train the best model")
    print("It has taken " + str(end3 - start1) + " seg to run")


def __main__():
    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    args = vars(ap.parse_args())
    # load the configuration and grab all image paths in the dataset
    fullAnalysis(args["conf"])

if __name__ == "__main__":
    __main__()


