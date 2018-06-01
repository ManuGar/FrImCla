# USAGE
# python train.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
import argparse
import cPickle
import h5py

from sklearn.model_selection import RandomizedSearchCV
import shallowmodels.classificationModelFactory as cmf

def train(outputPath, datasetPath, trainingSize):
	datasetName = datasetPath[datasetPath.rfind("/"):]
	auxPath = outputPath + datasetName

	file = open(auxPath + "/results/bestExtractorClassifier.csv")
	line =file.read()
	extractorClassifier = line.split(",")[0]
	extractor, classifier = extractorClassifier.split("_")

	labelEncoderPath = auxPath + "/models/le-" + extractor + ".cpickle"
	#[0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
	le = cPickle.loads(open(labelEncoderPath).read())
	# open the database and split the data into their respective training and
	# testing splits
	print("[INFO] gathering train/test splits...")

	featuresPath = auxPath + "/models/features-" + extractor + ".hdf5"
	#conf["features_path"][0:conf["features_path"].rfind(".")] + "-"+ conf["model"] +".hdf5"
	db = h5py.File(featuresPath)
	split = int(db["image_ids"].shape[0] * trainingSize)
	(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
	(testData, testLabels) = (db["features"][split:], db["image_ids"][split:])
	# use the label encoder to encode the training and testing labels
	trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]

	#It is not used
	#testLabels = [le.transform([l.split(":")[0]])[0] for l in testLabels]

	# define the grid of parameters to explore, then start the grid search where
	# we evaluate a Linear SVM for each value of C
	print("[INFO] tuning hyperparameters...")
	#file = open("testfile.text", "r")

	factory = cmf.classificationModelFactory()
	classifierModel = factory.getClassificationModel(classifier)
	model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
							   n_iter=classifierModel.getNIterations())

	model.fit(trainData, trainLabels)
	print("[INFO] best hyperparameters: {}".format(model.best_params_))
	# dump classifier to file
	print("[INFO] dumping classifier...")
	f = open(auxPath + "/classifier_" + extractor + "_" + classifier + ".cpickle", "w")
	f.write(cPickle.dumps(model))
	f.close()
	# close the database
	db.close()

def __main__():
	# construct the argument parser and parse the command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
	args = vars(ap.parse_args())
	# load the configuration and label encoder
	conf = Conf(args["conf"])

	outputPath=conf["output_path"]
	datasetPath=conf["dataset_path"]
	trainingSize=conf["training_size"]
	train(outputPath, datasetPath, trainingSize)

if __name__ == "__main__":
    __main__()