# USAGE
# python train.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.conf import Conf
from sklearn.neural_network import MLPClassifier
from utils import dataset
import numpy as np
import argparse
import cPickle
import h5py
from scipy.stats import randint as sp_randint

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and label encoder
conf = Conf(args["conf"])
labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
le = cPickle.loads(open(labelEncoderPath).read())

# open the database and split the data into their respective training and
# testing splits
print("[INFO] gathering train/test splits...")
featuresPath = conf["features_path"][0:conf["features_path"].rfind(".")] + "-"+ conf["model"] +".hdf5"
db = h5py.File(featuresPath)
split = int(db["image_ids"].shape[0] * conf["training_size"])
(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
(testData, testLabels) = (db["features"][split:], db["image_ids"][split:])

# use the label encoder to encode the training and testing labels
trainLabels = [le.transform([l.split(":")[0]])[0] for l in trainLabels]
testLabels = [le.transform([l.split(":")[0]])[0] for l in testLabels]



# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
if conf["modelClassifier"] == "RandomForest":
	param_grid = {"max_depth": [3, None],
			  "max_features": [1, 3, 10],
			  "min_samples_leaf": [1, 3, 10],
			  "bootstrap": [True, False],
			  "criterion": ["gini", "entropy"]}
	model = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid, cv=10)
elif conf["model"] == "LogisticRegression":
	params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
	model = GridSearchCV(LogisticRegression(), params, cv=10)
elif conf["modelClassifier"] == "GradientBoost":
	param_grid = {"max_depth": [3, None],
			  "max_features": [1, 3, 10],
			  "min_samples_leaf": [1, 3, 10]}
	model = GridSearchCV(GradientBoostingClassifier(n_estimators=20), param_grid, cv=10)
elif conf["modelClassifier"] == "KNN":
	param_grid = {'n_neighbors': range(5, 30,2)}
	model =  GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
elif conf["modelClassifier"] == "SVM":
	param_grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear']},  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
	model =  GridSearchCV(SVC(probability=True), param_grid, cv=10)
elif conf["modelClassifier"] == "MLP":
	param_grid = {'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs','sgd','adam'],
              'alpha': sp_randint(0.0001, 1),'learning_rate':['constant','invscaling','adaptive'],'momentum':[0.9,0.95,0.99]}
	model = GridSearchCV(MLPClassifier(random_state=84), param_grid, cv=10)


model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# # open the results file for writing and initialize the total number of accurate
# # rank-1 and rank-5 predictions
# print("[INFO] evaluating...")
# f = open(conf["results_path"] + conf["modelClassifier"] + ".txt", "w")
# rank1 = 0
# rank5 = 0
#
# # loop over the testing data
# for (label, features) in zip(testLabels, testData):
# 	# predict the probability of each class label and grab the top-5 labels
# 	# (based on probabiltiy)
# 	preds = model.predict_proba(np.atleast_2d(features))[0]
# 	preds = np.argsort(preds)[::-1][:5]
#
# 	# if the correct label if the first entry in the predicted labels list,
# 	# increment the number of correct rank-1 predictions
# 	if label == preds[0]:
# 		rank1 += 1
#
# 	# if the correct label is in the top-5 predicted labels, then increment
# 	# the number of correct rank-5 predictions
# 	if label in preds:
# 		rank5 += 1
#
# # convert the accuracies to percents and write them to file
# rank1 = (rank1 / float(len(testLabels))) * 100
# rank5 = (rank5 / float(len(testLabels))) * 100
# f.write("rank-1: {:.2f}%\n".format(rank1))
# f.write("rank-5: {:.2f}%\n\n".format(rank5))
#
# # write the classification report to file and close the output file
# predictions = model.predict(testData)
# f.write("{}\n".format(classification_report(testLabels, predictions,
# 	target_names=le.classes_)))
# f.write("Accuracy: {:.2f}%\n".format(accuracy_score(testLabels,predictions)))
# f.close()

# dump classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"] + conf["modelClassifiers"] + ".cpickle", "w")
f.write(cPickle.dumps(model))
f.close()

# close the database
db.close()