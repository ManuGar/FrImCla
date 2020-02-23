# import the necessary packages
#Example python prediction.py -c ../conf/confMelanomaDataAugmentation.conf -i ../pos.jpg -f overfeat -p [-3] -m svm
from __future__ import print_function
from __future__ import absolute_import

from frimcla.extractor.extractor import Extractor
from frimcla.utils.conf import Conf
from frimcla.utils import dataset
from frimcla.index_features import extractFeatures
from .shallowmodels.classificationModelFactory import classificationModelFactory
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import stats
import numpy as np
import argparse
try:
    import _pickle as cPickle
except ImportError:
	import cPickle
# import _pickle as cPickle
import h5py
import os
import json
import random
import sys


def prediction(featExt, classi, imagePath, outputPath, datasetPath,multiclass=False):
    # load the configuration, label encoder, and classifier
    print("[INFO] loading model...")
    datasetName = datasetPath[datasetPath.rfind("/")+1:]
    auxPath = outputPath + "/" + datasetName
    factory = classificationModelFactory()
    predictions = []

    with open(auxPath + "/ConfModel.json") as json_file:
        data = json.load(json_file)

    extractor = data['featureExtractors'][0]['model']
    classifier = data['featureExtractors'][0]['classificationModels'][0]

    if (featExt[0] == extractor and classi==classifier):
        labelEncoderPath = auxPath + "/models/le.cpickle"
        cPickleFile = auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle"
        le = cPickle.loads(open(labelEncoderPath, "rb").read())
        model = cPickle.loads(open(cPickleFile, "rb").read())

    else: #Aqui es donde deberia hacer la pregunta de que si quiere realmente entrenar ese modelo o el mejor
        print("This is not the best model. Are you sure you want to predict with it?")
        response = input()
        if (response.upper() in ("YES", "Y")):
            files = os.walk(auxPath + "/models")
            listFeatExt = []
            for _, _, file in files:
                for feat in file:
                    if(len(feat.split("-"))>1):
                        feat = feat.split("-")[1].split(".")[0]
                        listFeatExt.append(feat)
            listFeatExt = list(set(listFeatExt))
            if (featExt[0] in listFeatExt):
                labelEncoderPath = auxPath + "/models/le.cpickle"
                le = cPickle.loads(open(labelEncoderPath,"rb").read())
                featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
                db = h5py.File(featuresPath)
                split = int(db["image_ids"].shape[0])
                (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
                trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
                classifierModel = factory.getClassificationModel(classi)

                model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                           n_iter=classifierModel.getNIterations())
                model.fit(trainData, trainLabels)
                f = open(auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle", "wb")
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
                labelEncoderPath = auxPath + "/models/le.cpickle"
                le = cPickle.loads(open(labelEncoderPath,"rb").read())
                featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
                db = h5py.File(featuresPath)
                split = int(db["image_ids"].shape[0])
                (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
                trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
                classifierModel = factory.getClassificationModel(classi)
                model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                           n_iter=classifierModel.getNIterations())
                model.fit(trainData, trainLabels)
                f = open(auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle", "wb")
                f.write(cPickle.dumps(model))
                f.close()
                db.close()
        else:
            labelEncoderPath = auxPath + "/models/le.cpickle"
            with open(auxPath + "/ConfModel.json") as json_file:
                data = json.load(json_file)
            extractor = data['featureExtractors'][0]['model']
            classifier = data['featureExtractors'][0]['classificationModels'][0]
            cPickleFile = auxPath + "/classifiers/classifier_" + extractor + "_" + classifier + ".cpickle"
            le = cPickle.loads(open(labelEncoderPath, "rb").read())
            model = cPickle.loads(open(cPickleFile, "rb").read())

    filePrediction = open(auxPath+"/predictionResults.csv","a")
    filePrediction.write("image_id, " + datasetName +"\n")
    oe = Extractor(featExt)
    imagePaths = list(paths.list_images(imagePath))

    if (len(imagePaths)==0):
        (labels, images) = dataset.build_batch([imagePath], featExt[0])
        features = oe.describe(images)
        for (label, vector) in zip(labels, features):
            prediction = model.predict(np.atleast_2d(vector))[0]
            filePrediction.write(str(label) + ", " + str(prediction) + "\r\n")
            print(prediction)
            if multiclass:
                prediction = le.inverse_transform(np.array([prediction]))
            else:
                prediction = le.inverse_transform([prediction])
            predictions.append(prediction)
            print("[INFO] class predicted for the image {}: {}".format(label, prediction))
    else:
        for (i, imPaths) in enumerate(dataset.chunk(imagePaths, 32)):
            (labels, images) = dataset.build_batch(imPaths, featExt[0])
            features = oe.describe(images)
            for (label, vector) in zip(labels, features):
                prediction = model.predict(np.atleast_2d(vector))[0]
                if multiclass:
                    prediction = le.inverse_transform(np.array([prediction]))
                else:
                    prediction = le.inverse_transform([prediction])
                predictions.append(prediction)
                filePrediction.write( str(label) + ", " + str(prediction) + "\r\n")
                print("[INFO] class predicted for the image {}: {}".format(label, prediction))

    filePrediction.close()
    return predictions










def predictionArray(featExt, classi, imagesPredict, outputPath, datasetPath,multiclass=False):
    # load the configuration, label encoder, and classifier
    print("[INFO] loading model...")
    datasetName = datasetPath[datasetPath.rfind("/")+1:]
    auxPath = outputPath + "/" + datasetName
    factory = classificationModelFactory()
    predictions = []

    with open(auxPath + "/ConfModel.json") as json_file:
        data = json.load(json_file)

    extractor = data['featureExtractors'][0]['model']
    classifier = data['featureExtractors'][0]['classificationModels'][0]

    if (featExt[0] == extractor and classi==classifier):
        labelEncoderPath = auxPath + "/models/le.cpickle"
        cPickleFile = auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle"
        le = cPickle.loads(open(labelEncoderPath, "rb").read())
        model = cPickle.loads(open(cPickleFile, "rb").read())

    else: #Aqui es donde deberia hacer la pregunta de que si quiere realmente entrenar ese modelo o el mejor
        print("This is not the best model. Are you sure you want to predict with it?")
        response = input()
        if (response.upper() in ("YES", "Y")):
            files = os.walk(auxPath + "/models")
            listFeatExt = []
            for _, _, file in files:
                for feat in file:
                    if(len(feat.split("-"))>1):
                        feat = feat.split("-")[1].split(".")[0]
                        listFeatExt.append(feat)
            listFeatExt = list(set(listFeatExt))
            if (featExt[0] in listFeatExt):
                labelEncoderPath = auxPath + "/models/le.cpickle"
                le = cPickle.loads(open(labelEncoderPath,"rb").read())
                featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
                db = h5py.File(featuresPath)
                split = int(db["image_ids"].shape[0])
                (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
                trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
                classifierModel = factory.getClassificationModel(classi)

                model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                           n_iter=classifierModel.getNIterations())
                model.fit(trainData, trainLabels)
                f = open(auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle", "wb")
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
                labelEncoderPath = auxPath + "/models/le.cpickle"
                le = cPickle.loads(open(labelEncoderPath,"rb").read())
                featuresPath = auxPath + "/models/features-" + featExt[0] + ".hdf5"
                db = h5py.File(featuresPath)
                split = int(db["image_ids"].shape[0])
                (trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
                trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
                classifierModel = factory.getClassificationModel(classi)
                model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
                                           n_iter=classifierModel.getNIterations())
                model.fit(trainData, trainLabels)
                f = open(auxPath + "/classifiers/classifier_" + featExt[0] + "_" + classi + ".cpickle", "wb")
                f.write(cPickle.dumps(model))
                f.close()
                db.close()
        else:
            labelEncoderPath = auxPath + "/models/le.cpickle"
            with open(auxPath + "/ConfModel.json") as json_file:
                data = json.load(json_file)
            extractor = data['featureExtractors'][0]['model']
            classifier = data['featureExtractors'][0]['classificationModels'][0]
            cPickleFile = auxPath + "/classifiers/classifier_" + extractor + "_" + classifier + ".cpickle"
            le = cPickle.loads(open(labelEncoderPath, "rb").read())
            model = cPickle.loads(open(cPickleFile, "rb").read())

    filePrediction = open(auxPath+"/predictionResults.csv","a")
    filePrediction.write("image_id, " + datasetName +"\n")
    oe = Extractor(featExt)

    for (i, imPaths) in enumerate(dataset.chunk(imagesPredict, 32)):
        (labels, images) = dataset.build_batch(imPaths, featExt[0])
        features = oe.describe(images)
        for (label, vector) in zip(labels, features):
            prediction = model.predict(np.atleast_2d(vector))[0]
            if multiclass:
                prediction = le.inverse_transform(np.array([prediction]))
            else:
                prediction = le.inverse_transform([prediction])
            predictions.append(prediction)
            filePrediction.write( str(label) + ", " + str(prediction) + "\r\n")
            print("[INFO] class predicted for the image {}: {}".format(label, prediction))

    filePrediction.close()
    return predictions














#This method only if you want to execute the prediction with the output of the previous steps
def predictionSimple(imagePath, outputPath, datasetPath):
    datasetName = datasetPath[datasetPath.rfind("/"):]
    auxPath = outputPath + datasetName
    with open(auxPath + "ConfModel.json") as json_file:
        data = json.load(json_file)

    extractor = data[0]['featureExtractor']
    classifier = data[0]['classifierModel']
    prediction(extractor,classifier,imagePath,outputPath,datasetPath)

def predictionEnsemble(featureExtractors, classifiers, imagePath, outputPath, datasetName):
    auxPath = outputPath + "/" + datasetName
    filePrediction = open(auxPath + "/predictionResults.csv", "a")
    filePrediction.write("image_id, " + datasetName)
    labelEncoderPath = auxPath + "/models/le.cpickle"
    le = cPickle.loads(open(labelEncoderPath,"rb").read())
    imagePaths = list(paths.list_images(imagePath))
    predictions = []
    modes = []
    predictionsout = []

    for fe in featureExtractors:
        if (len(imagePaths)==0):
            (labels, images) = dataset.build_batch([imagePath], fe[0])
        else:
            (labels, images) = dataset.build_batch(imagePaths, fe[0])
        oe = Extractor(fe)
        features = oe.describe(images)
        for classi in classifiers:
            cPickleFile = auxPath + "/classifiers/classifier_" + fe[0] + "_" + classi + ".cpickle"
            if os.path.isfile(cPickleFile):
                model = cPickle.loads(open(cPickleFile,"rb").read())
                prediction = model.predict(features)
                predictions.append(prediction)
    # aux = np.array(predictions)
    aux = []
    for i in range((len(imagePaths))):
        for x in predictions:
            aux.append(x[i])
        aux = np.array(aux)
        mode = stats.mode(aux[0])

        # mode = le.inverse_transform(mode[0])
        modes.append(mode[0])
        aux = []

    # mode = stats.mode(aux[0])
    # prediction = le.inverse_transform(mode[0])
    # filePrediction.write(str(labels) + ", " + str(prediction) + "\r\n")
    # le = cPickle.loads(open(labelEncoderPath).read())
    for (image, mode) in zip(imagePaths,modes):
        prediction = le.inverse_transform(mode)
        print("[INFO] class predicted for the image {}: {}".format(image, prediction[0]))
        filePrediction.write("\n" + str(image) + ", " + str(prediction[0]))
        predictionsout.append(prediction)

    filePrediction.close()
    return predictionsout

def __main__():
    # construct the argument parser and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the image to predict")
    ap.add_argument("-f", "--features", required=True, help="feature extractor to predict the class")
    ap.add_argument("-m", "--classifierModels", required=True, help="model to classify the images")
    ap.add_argument("-o", "--outputPath", required=True, help="Path of the output of the algorithm")
    ap.add_argument("-d", "--datasetPath", required=True, help="model to classify the images")

    args = vars(ap.parse_args())
    conf = Conf(args["conf"])
    outputPath = conf["output_path"]
    featureExtractors = args["feature"]
    classifiers = args["classifierModels"]
    datasetPath = conf["dataset_path"]
    # This method must have the same output path and dataset path that in the other parts of the program, in other case the program
    # will not work
    # prediction(featureExtractors, classifiers, args["image"], outputPath, datasetPath)
    predictionEnsemble(featureExtractors, classifiers, args["image"], outputPath, datasetPath)


if __name__ == "__main__":
    __main__()
