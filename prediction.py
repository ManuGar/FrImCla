# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
from extractor.extractor import Extractor
import numpy as np
import argparse
from utils.conf import Conf

from utils import dataset

import cPickle
import imutils

def prediction(featExt,imagePath, outputPath, datasetPath):
    # load the configuration, label encoder, and classifier
    print("[INFO] loading model...")
    datasetName = datasetPath[datasetPath.rfind("/"):]
    file = open(outputPath + datasetName + "/results/bestExtractorClassifier.csv")
    fParams = open ("featureExtractors.csv")
    for line in fParams:
        fExt,fPar = line.split(",")[0:]
        print (fExt)
        print ("EEEEEEEEEEEEEEEEEEEEEEEEE")
        print (fParams)
        print ("AAAAAAAAAAAAAAAAAAAAAAAAA")
        if fExt == featExt[0] and fParams == featExt[1:]:
            pass
        else:
            pass

    line = file.read()
    extractorClassifier = line.split(",")[0]
    extractor, classifier = extractorClassifier.split("_")
    labelEncoderPath = outputPath + datasetName + "/models/le-" + extractor + ".cpickle"



    # labelEncoderPath =conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
    le = cPickle.loads(open(labelEncoderPath).read())
    cPickleFile = outputPath + datasetName + "/classifier_" + extractor + "_" + classifier + ".cpickle"
    model = cPickle.loads(open(cPickleFile).read())
    # open(conf["classifier_path"]+ conf["modelClassifier"] + ".cpickle").read()

    #imagePath = args["image"]
    # for featureExtractor in conf["featureExtractors"]:
    #     if extractor == featureExtractor[0]:
    #         oe=Extractor(featureExtractor)
    #         break







#todo A esto le falta pasarle los parametros para que se cree el extractor correctamente. Buscarlo en lo que hemos
#todo ejecutado antes o preguntar al usuario. Si preguntamos al usuario, si no coincide con el mejor avisarlo y hacer que
#todo decida él o no se como hacerlo (habrá que buscar los datos del mejor modelo en todos los archivos anteriores)





    oe = Extractor([extractor])
    (labels, images) = dataset.build_batch([imagePath], extractor)  # conf["model"]

    features = oe.describe(images)
    for (label, vector) in zip(labels, features):
        prediction = model.predict(np.atleast_2d(vector))[0]
        print(prediction)
        prediction = le.inverse_transform(prediction)
        print("[INFO] predicted: {}".format(prediction))

def __main__():
    # construct the argument parser and parse the command line arguments
    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    ap.add_argument("-i", "--image", required=True, help="path to the image to predict")
    ap.add_argument("-f", "--feature", required=True, help="feature extractor to predict the class")

    # ap.add_argument("-u", "--control", required=False, help="use control image")
    # ap.add_argument("-l", "--controlImage", required=False, help="path to the control image")
    args = vars(ap.parse_args())
    conf = Conf(args["conf"])
    outputPath = conf["output_path"]
    featureExtractor = args["feature"]

    datasetPath = conf["dataset_path"]
    # load the configuration and grab all image paths in the dataset
	# generateFeatures(conf,imagePaths,"False")
    prediction(featureExtractor, args["image"],outputPath, datasetPath )


if __name__ == "__main__":
    __main__()


#
# # load the configuration, label encoder, and classifier
# print("[INFO] loading model...")
# conf = Conf(args["conf"])
#
#
# datasetName = conf["dataset_path"][conf["dataset_path"].rfind("/"):]
#
# file = open(conf["output_path"] + datasetName + "/results/bestExtractorClassifier.csv")
# line =file.read()
# extractorClassifier = line.split(",")[0]
# extractor, classifier = extractorClassifier.split("_")
# labelEncoderPath = conf["output_path"]+ datasetName + "/models/le-" + extractor + ".cpickle"
#
# #labelEncoderPath =conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
# le = cPickle.loads(open(labelEncoderPath).read())
# cPickleFile = conf["output_path"] + datasetName + "/classifier_" + extractor + "_" + classifier + ".cpickle"
# model = cPickle.loads(open(cPickleFile).read())
# #open(conf["classifier_path"]+ conf["modelClassifier"] + ".cpickle").read()
#
# imagePath = args["image"]
# # for featureExtractor in conf["featureExtractors"]:
# #     if extractor == featureExtractor[0]:
# #         oe=Extractor(featureExtractor)
# #         break
# oe = Extractor(extractor)
# (labels, images) = dataset.build_batch([imagePath], extractor) #conf["model"]
#
# features = oe.describe(images)
# for (label, vector) in zip(labels, features):
#     prediction = model.predict(np.atleast_2d(vector))[0]
#     print(prediction)
#     prediction = le.inverse_transform(prediction)
#     print("[INFO] predicted: {}".format(prediction))