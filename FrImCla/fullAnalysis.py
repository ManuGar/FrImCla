# USAGE
# python fullAnalysis.py --conf conf/flowers17.json
# import the necessary packages
from __future__ import print_function
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import argparse
import shutil
from utils.conf import Conf
from index_features import generateFeatures
from StatisticalComparison import statisticalComparison
from train import train
import json
import zipfile,os
# suppress any FutureWarning from Theano


def fullAnalysis(config):
    conf = Conf(config)
    verbose = False
    start = time.time()

    aux = conf["output_path"] + conf["dataset_path"][conf["dataset_path"].rfind("/"):]
    generateFeatures(conf["output_path"], conf["batch_size"], conf["dataset_path"], conf["feature_extractors"], verbose)
    statisticalComparison(conf["output_path"], conf["dataset_path"], conf["feature_extractors"],
                          conf["model_classifiers"], conf["measure"], verbose)

    end = time.time()
    train(conf["output_path"], conf["dataset_path"], conf["training_size"])

    print("It has taken " + str(end - start) + " seg to run")
    print("Do you want to generate a web app to classify the images with the best combination? y/n")
    webapp = raw_input()
    with open(aux + "/ConfModel.json") as json_file:
        data = json.load(json_file)
    extractor = data['featureExtractor']
    classifier = data['classificationModel']
    if(webapp.upper() in ("YES","Y")):
        shutil.copyfile(aux + "/ConfModel.json", "./webApp/FlaskApp/ConfModel.json")
        shutil.copyfile(aux + "/classifier_" + extractor["model"] + "_" + classifier + ".cpickle", "./webApp/FlaskApp/classifier_" + extractor["model"] + "_" + classifier + ".cpickle")
        shutil.copyfile(aux + "/models/le-" + extractor["model"] + ".cpickle", "./webApp/FlaskApp/le-" + extractor["model"] + ".cpickle")
        shutil.make_archive(aux + "/webapp", 'zip', './webApp')

def __main__():
    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    args = vars(ap.parse_args())
    # load the configuration and grab all image paths in the dataset
    fullAnalysis(args["conf"])

if __name__ == "__main__":
    __main__()


