# USAGE
# python fullAnalysis.py --conf conf/flowers17.json
# import the necessary packages


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import os
from frimcla.utils.conf import Conf
from frimcla.index_features import generateFeatures
from frimcla.StatisticalComparison import statisticalComparison, majorityVoting
from frimcla.train import train
# suppress any FutureWarning from Theano



"""
    With this method we execute the whole process to obtain the model to predict the class that the images belongs to. This process has three parts:
    generation of features, creation of the prediction model (in some cases there is a statistical analisys to obtain the comparison of the models and decide the best one) 
    and training of the model choosing in the previous step.

"""
def fullAnalysis(config):
    featureExtractors = [["vgg16", "False"], ["vgg19", "False"],["resnet", "False"], ["inception", "False"],["googlenet"], ["overfeat", "[-3]"], ["xception", "False"],
                         ["densenet"], ["lab888"], ["lab444","4,4,4"], ["hsv888"], ["hsv444","4,4,4"], ["haralick"], ["hog"], ["haarhog"]]

    # ["mymodel"],
    modelClassifiers = ["GradientBoost","RandomForest", "SVM","KNN","LogisticRegression", "MLP"]
    conf = Conf(config)
    verbose = False
    if (conf["expert"]):
        featureExtractors = conf["feature_extractors"]
        modelClassifiers = conf["model_classifiers"]
    geneFeatuT =generateFeatures(conf["output_path"], conf["batch_size"], conf["dataset_path"], featureExtractors, verbose)

    if(conf["ensemble"]):
        comparisonT = majorityVoting(conf["output_path"], conf["dataset_path"], featureExtractors,
                       modelClassifiers, conf["measure"], verbose)
    else:
        comparisonT = statisticalComparison(conf["output_path"], conf["dataset_path"], featureExtractors,
                              modelClassifiers, conf["measure"], conf["n_steps"], verbose)

    trainT = train(conf["output_path"], conf["dataset_path"], conf["training_size"])
    dataset = conf["dataset_path"][conf["dataset_path"].rfind("/"):]

    f = open(os.path.abspath(conf["output_path"]+ dataset + "/timeFile.txt"), "w")
    f.write("It has taken " + geneFeatuT + " seg to generate the features\n")
    f.write("It has taken " + comparisonT + " seg to generate the model comparison and the statistical analysis\n")
    f.write("It has taken " + trainT + " seg to train the best model\n")
    totalT = geneFeatuT + comparisonT + trainT
    f.write("It has taken " + str(totalT) + " seg to run\n")
    print("\nIt has taken " + geneFeatuT + " seg to generate the features")
    print("It has taken " + comparisonT + " seg to generate the model comparison and the statistical analysis")
    print("It has taken " + trainT + " seg to train the best model")
    print("It has taken " + totalT + " seg to run")


def __main__():

    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    args = vars(ap.parse_args())
    # load the configuration and grab all image paths in the dataset
    fullAnalysis(args["conf"])

if __name__ == "__main__":
    __main__()


