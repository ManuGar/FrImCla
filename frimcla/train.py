# USAGE
# python train.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
import argparse
import cPickle
import h5py
import shutil
import json

from sklearn.model_selection import RandomizedSearchCV
import shallowmodels.classificationModelFactory as cmf


"""
	This algorithm trains the method indicated in the ConfModel.json file. In this file is sotred the feature extractor,
	the params of the extractor and the classifier model. The method returns the model trained and asks to the user if
	he wants a webapp.  The web application is a simple application that allows the user to exploit the model to predict
	the classes of the images.
"""
def train(outputPath, datasetPath, trainingSize):
	datasetName = datasetPath[datasetPath.rfind("/"):]
	auxPath = outputPath + datasetName
	with open(auxPath + "/ConfModel.json") as json_file:
		data = json.load(json_file)
	extractor = data['featureExtractor']
	classifier = data['classificationModel']
	labelEncoderPath = auxPath + "/models/le-" + extractor["model"] + ".cpickle"
	le = cPickle.loads(open(labelEncoderPath).read())
	# open the database and split the data into their respective training and
	# testing splits
	print("[INFO] gathering train/test splits...")
	featuresPath = auxPath + "/models/features-" + extractor["model"] + ".hdf5"
	db = h5py.File(featuresPath)
	split = int(db["image_ids"].shape[0] * trainingSize)
	(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
	# use the label encoder to encode the training and testing labels
	trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
	# define the grid of parameters to explore, then start the grid search where
	# we evaluate a Linear SVM for each value of C
	print("[INFO] tuning hyperparameters...")
	factory = cmf.classificationModelFactory()
	classifierModel = factory.getClassificationModel(classifier)
	model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
							   n_iter=classifierModel.getNIterations())

	model.fit(trainData, trainLabels)
	print("[INFO] best hyperparameters: {}".format(model.best_params_))
	# dump classifier to file
	print("[INFO] dumping classifier...")
	f = open(auxPath + "/classifier_" + extractor["model"] + "_" + classifier + ".cpickle", "w")
	f.write(cPickle.dumps(model))
	f.close()
	# close the database
	db.close()
	print("Do you want to generate a web app to classify the images with the best combination? y/n")
	webapp = raw_input()
	with open(auxPath + "/ConfModel.json") as json_file:
		data = json.load(json_file)
	extractor = data['featureExtractor']
	classifier = data['classificationModel']
	if (webapp.upper() in ("YES", "Y")):
		shutil.copytree("./frimcla/webApp/", auxPath + "/webApp")
		shutil.copyfile(auxPath + "/ConfModel.json", auxPath +"/webApp/FlaskApp/ConfModel.json")
		shutil.copyfile(auxPath + "/classifier_" + extractor["model"] + "_" + classifier + ".cpickle",
						auxPath + "/webApp/FlaskApp/classifier_" + extractor["model"] + "_" + classifier + ".cpickle")
		shutil.copyfile(auxPath + "/models/le-" + extractor["model"] + ".cpickle",
						auxPath + "/webApp/FlaskApp/le-" + extractor["model"] + ".cpickle")
		shutil.make_archive(auxPath + "/webApp", 'zip', auxPath + '/webApp')
		shutil.rmtree(auxPath + '/webApp')

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