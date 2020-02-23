# USAGE
# python train.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from __future__ import absolute_import
from sklearn.model_selection import RandomizedSearchCV
import argparse
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import h5py
import shutil
import json
import os
# from .utils.conf import Conf
from frimcla.utils.conf import Conf
from frimcla.shallowmodels import classificationModelFactory as cmf
from frimcla.shallowmodels import classificationModelMultiClassFactory as cmmcf
# from . import shallowmodels.classificationModelFactory as cmf
import wget
import zipfile
import time
import re



"""
	This algorithm trains the method indicated in the ConfModel.json file. In this file is sotred the feature extractor,
	the params of the extractor and the classifier model. The method returns the model trained and asks to the user if
	he wants a webapp.  The web application is a simple application that allows the user to exploit the model to predict
	the classes of the images.
	The method returns the execution time.
"""
def train(outputPath, datasetPath, trainingSize,multiclass=False):
	start = time.time()
	datasetName = datasetPath[datasetPath.rfind("/"):]
	auxPath = outputPath + datasetName
	with open(auxPath + "/ConfModel.json") as json_file:
		data = json.load(json_file)

	extractors = data['featureExtractors']
	# classifiers = data['classificationModels']
	for ex in extractors:
		labelEncoderPath = auxPath + "/models/le.cpickle"
		le = cPickle.loads(open(labelEncoderPath, "rb").read())
		# open the database and split the data into their respective training and
		# testing splits
		print("[INFO] gathering train/test splits...")
		featuresPath = auxPath + "/models/features-" + ex["model"] + ".hdf5"
		db = h5py.File(featuresPath)
		split = int(db["image_ids"].shape[0] * trainingSize)
		(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
		# use the label encoder to encode the training and testing labels

		# trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
		# define the grid of parameters to explore, then start the grid search where
		# we evaluate a Linear SVM for each value of C
		print("[INFO] tuning hyperparameters...")
		if multiclass:
			factory = cmmcf.classificationModelMultiClassFactory()
			import numpy as np
			trainLabels = np.array([list(le.transform([re.split(":|\\\\", l)[-2].split('_')])[0]) for l in trainLabels])
		else:
			factory = cmf.classificationModelFactory()
			trainLabels = [le.transform([re.split(":|\\\\", l)[-2]])[0] for l in trainLabels]
		for clas in ex["classificationModels"]:
			classifierModel = factory.getClassificationModel(clas)
			model = RandomizedSearchCV(classifierModel.getModel(), param_distributions=classifierModel.getParams(),
							   n_iter=classifierModel.getNIterations())


			model.fit(trainData, trainLabels)
			print("[INFO] best hyperparameters: {}".format(model.best_params_))
			# dump classifier to file
			print("[INFO] dumping classifier...")
			if (not os.path.exists(auxPath + "/classifiers")):
				os.makedirs(auxPath + "/classifiers")
			f = open(auxPath + "/classifiers/classifier_" + ex["model"] + "_" + clas + ".cpickle", "wb")
			f.write(cPickle.dumps(model))
			f.close()
			# close the database
			db.close()
	finish = time.time()
	print("Do you want to generate a web app to classify the images with the best combination? y/n")
	webapp = input()
	# with open(auxPath + "/ConfModel.json") as json_file:
	# 	data = json.load(json_file)
	# extractor = data['featureExtractor']
	# classifier = data['classificationModel']

	if (webapp.upper() in ("YES", "Y")):
		url = "https://drive.google.com/uc?id=1r_Dk4dhVq0ABVf5YeCTBjbLAdl61KkTn&export=download&authuser=0"
		file = wget.download(url, auxPath + "/webApp.zip")
		zip = zipfile.ZipFile(file)#auxPath + '/webApp.zip'
		zip.extractall(auxPath)
		# shutil.copytree("./frimcla/webApp/", auxPath + "/webApp")
		shutil.copyfile(auxPath + "/ConfModel.json", auxPath + "/webApp/FlaskApp/ConfModel.json")
		for ext in extractors:
			for classi in ext["classificationModels"]:
				shutil.copyfile(auxPath + "/classifiers/classifier_" + ext["model"] + "_" + classi + ".cpickle",
								auxPath + "/webApp/FlaskApp/classifiers/classifier_" + ext["model"] + "_" + classi + ".cpickle")
				shutil.copyfile(auxPath + "/models/le.cpickle",
								auxPath + "/webApp/FlaskApp/le.cpickle")

		zip.close()
		os.remove(auxPath + '/webApp.zip')
		shutil.make_archive(auxPath + "/webApp", 'zip', auxPath + '/webApp')
		shutil.rmtree(auxPath + '/webApp')
	return finish - start

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