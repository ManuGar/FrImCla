# import the necessary packages
#Example python prediction.py -c ../conf/confMelanomaDataAugmentation.conf -i ../pos.jpg -f overfeat -p [-3] -m svm
from __future__ import print_function

from extractor.extractor import Extractor
from utils.conf import Conf
from utils import dataset
from index_features import extractFeatures
from shallowmodels.classificationModelFactory import classificationModelFactory
import numpy as np
import argparse
import cPickle
import h5py
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import random
from imutils import paths
import os
import json

def prediction(featExt, classi, imagePath, outputPath, datasetPath):
    # load the configuration, label encoder, and classifier
    print("[INFO] loading model...")
    datasetName = datasetPath[datasetPath.rfind("/"):]
    auxPath = outputPath + datasetName
    factory = classificationModelFactory()

    cPickleFile = auxPath + "/classifier_" + featExt[0] + "_" + classi + ".cpickle"
    if os.path.isfile(cPickleFile):
        labelEncoderPath = auxPath + "/models/le-" + featExt[0] + ".cpickle"
        le = cPickle.loads(open(labelEncoderPath).read())
        # cPickleFile = auxPath + "/classifier_" + extractor + "_" + classifier + ".cpickle"
        model = cPickle.loads(open(cPickleFile).read())
        # open(conf["classifier_path"]+ conf["modelClassifier"] + ".cpickle").read()

    else:
        files = os.walk(auxPath + "/models")
        listFeatExt = []
        for  _, _, file in files:
            for feat in file:
                feat = feat.split("-")[1].split(".")[0]
                listFeatExt.append(feat)
        listFeatExt = list(set(listFeatExt))
        if(featExt[0] in listFeatExt):
            labelEncoderPath = auxPath + "/models/le-" + featExt[0] + ".cpickle"
            le = cPickle.loads(open(labelEncoderPath).read())
            featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
            db = h5py.File(featuresPath)
            split = int(db["image_ids"].shape[0])
            (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
            trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
            classifierModel = factory.getClassificationModel(classi)

            model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                       n_iter=classifierModel.getNIterations())
            model.fit(trainData, trainLabels)
            f = open(auxPath + "/classifier_" + featExt[0] + "_" + classi + ".cpickle", "w")
            f.write(cPickle.dumps(model))
            f.close()
            db.close()

        else:
            imagePaths = list(paths.list_images(datasetPath))
            random.seed(42)
            random.shuffle(imagePaths)
            le = LabelEncoder()
            le.fit([p.split("/")[-2] for p in imagePaths])
            extractFeatures(featExt, 32, imagePaths, outputPath, datasetPath, le, False)
            labelEncoderPath = auxPath + "/models/le-" + featExt[0] + ".cpickle"
            le = cPickle.loads(open(labelEncoderPath).read())
            featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
            db = h5py.File(featuresPath)
            split = int(db["image_ids"].shape[0])
            (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
            trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
            classifierModel = factory.getClassificationModel(classi)
            model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                           n_iter=classifierModel.getNIterations())
            model.fit(trainData, trainLabels)
            f = open(auxPath + "/classifier_" + featExt[0] + "_" + classi + ".cpickle", "w")
            f.write(cPickle.dumps(model))
            f.close()
            db.close()


    filePrediction = open(auxPath+"/predictionResults.csv","a")
    filePrediction.write("image_id, melanoma\n")
    oe = Extractor(featExt)
    imagePaths = list(paths.list_images(imagePath))

    for (i, imPaths) in enumerate(dataset.chunk(imagePaths, 32)):
        (labels, images) = dataset.build_batch(imPaths, featExt[0])

        features = oe.describe(images)
        for (label, vector) in zip(labels, features):
            prediction = model.predict(np.atleast_2d(vector))[0]
            filePrediction.write( str(label) + ", " + str(prediction) + "\r\n")
            prediction = le.inverse_transform(prediction)
            print("[INFO] class predicted for the image {}: {}".format(label, prediction))

    filePrediction.close()


#This method only if you want to execute the prediction with the output of the previous steps
def predictionSimple(imagePath, outputPath, datasetPath):
    datasetName = datasetPath[datasetPath.rfind("/"):]
    auxPath = outputPath + datasetName
    with open(auxPath + "ConfModel.json") as json_file:
        data = json.load(json_file)

    extractor = data[0]['featureExtractor']
    classifier = data[0]['classifierModel']
    prediction(extractor,classifier,imagePath,outputPath,datasetPath)

def __main__():
    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    ap.add_argument("-i", "--image", required=True, help="path to the image to predict")
    ap.add_argument("-f", "--feature", required=True, help="feature extractor to predict the class")
    ap.add_argument("-p", "--params", required=False , help="parameters of the feature extractor")
    ap.add_argument("-m", "--classifierModel", required=True, help="model to classify the images")

    args = vars(ap.parse_args())
    conf = Conf(args["conf"])
    outputPath = conf["output_path"]
    featureExtractor = [args["feature"], args["params"]]
    classifier = args["classifierModel"]

    datasetPath = conf["dataset_path"]
    # load the configuration and grab all image paths in the dataset
    # generateFeatures(conf,imagePaths,"False")
    prediction(featureExtractor, classifier, args["image"], outputPath, datasetPath)


if __name__ == "__main__":
    __main__()
