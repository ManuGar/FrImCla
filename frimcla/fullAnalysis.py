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
from StatisticalComparison import statisticalComparison, majorityVoting
from train import train
# suppress any FutureWarning from Theano



"""
    With this method we execute the whole process to obtain the model to predict the class that the images belongs to. This process has three parts:
    generation of features, creation of the prediction model (in some cases there is a statistical analisys to obtain the comparison of the models and decide the best one) 
    and training of the model choosing in the previous step.

"""
def fullAnalysis(config):
    featureExtractors = [["mymodel"], ["vgg16", "False"], ["vgg19", "False"],["resnet", "False"], ["inception", "False"],["googlenet"], ["overfeat", "[-3]"], ["xception", "False"],
                         ["densenet"], ["lab888"], ["lab444","4,4,4"], ["hsv888"], ["hsv444","4,4,4"], ["haralick"], ["hog"], ["haarhog"]]
    modelClassifiers = ["GradientBoost","RandomForest", "SVM","KNN","LogisticRegression", "MLP"]
    conf = Conf(config)
    verbose = False
    start1 = time.time()
    if (conf["expert"]):
        featureExtractors = conf["feature_extractors"]
        modelClassifiers = conf["model_classifiers"]
    generateFeatures(conf["output_path"], conf["batch_size"], conf["dataset_path"], featureExtractors, verbose)
    end1 = time.time()
    start2 = time.time()

    if(conf["ensemble"]):
        majorityVoting(conf["output_path"], conf["dataset_path"], featureExtractors,
                       modelClassifiers, conf["measure"], verbose)
    else:
        statisticalComparison(conf["output_path"], conf["dataset_path"], featureExtractors,
                              modelClassifiers, conf["measure"], conf["n_steps"], verbose)

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
    print("\nIt has taken " + str(end1 - start1) + " seg to generate the features")
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


