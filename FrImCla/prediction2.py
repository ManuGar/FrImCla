# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
from FrImCla.extractor.extractor import Extractor
import numpy as np
import argparse
from utils import dataset

import cPickle
import cv2
import imutils


def combineImages(controlPath,imagePath):
    control = cv2.imread(controlPath)
    image = cv2.imread(imagePath)
    control = imutils.resize(control,height=300)
    image = imutils.resize(image,height=300)
    return np.hstack([control,image])


# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to predict")
ap.add_argument("-u", "--control", required=False, help="use control image")
ap.add_argument("-l", "--controlImage", required=False, help="path to the control image")

args = vars(ap.parse_args())

# load the configuration, label encoder, and classifier
print("[INFO] loading model...")
conf = Conf(args["conf"])
labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
le = cPickle.loads(open(labelEncoderPath).read())
model = cPickle.loads(open(conf["classifier_path"]+ conf["model_classifier"] + ".cpickle").read())

imagePath = args["image"]
oe = Extractor(conf["model"])
if(args["control"]=="True"):
    print("Control")
    controlPath = args["controlImage"]
    cv2.imwrite("temp.jpg",combineImages(controlPath,imagePath))
    (labels, images) = dataset.build_batch(["temp.jpg"], conf["model"])
else:
    (labels, images) = dataset.build_batch([imagePath], conf["model"])


features = oe.describe(images)
for (label, vector) in zip(labels, features):
    prediction = model.predict(np.atleast_2d(vector))[0]
    print(prediction)
    prediction = le.inverse_transform(prediction)
    print("[INFO] predicted: {}".format(prediction))