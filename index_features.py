# USAGE
# python index_features.py --conf conf/flowers17.json

# suppress any FutureWarning from Theano
from __future__ import print_function
from mpi4py import MPI
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# import the necessary packages
from extractor.extractor import Extractor
from indexer.indexer import Indexer
from utils.conf import Conf
from utils import dataset
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import argparse
import cPickle
import random
import os
from sklearn.externals.joblib import Parallel, delayed


def extractFeatures(fE, batchSize, datasetPath, outputPath, datasetP, le, verbose):
	# initialize the Overfeat extractor and the Overfeat indexer
	if verbose:
		print("[INFO] initializing network...")
	oe = Extractor(fE)

	featuresPath = outputPath + datasetP[datasetP.rfind("/"):] + \
				   "/models/features-" + fE[0] + ".hdf5"
	directory = featuresPath[:featuresPath.rfind("/")]
	if (not os.path.exists(directory)):
		os.makedirs(directory)
	oi = Indexer(featuresPath, estNumImages=len(datasetPath))
	if verbose:
		print("[INFO] starting feature extraction...")

	# loop over the image paths in batches
	for (i, paths) in enumerate(dataset.chunk(datasetPath, batchSize)):
		# load the set of images from disk and describe them
		(labels, images) = dataset.build_batch(paths, fE[0])
		features = oe.describe(images)

		# loop over each set of (label, vector) pair and add them to the indexer
		for (label, vector) in zip(labels, features):
			oi.add(label, vector)

		# check to see if progress should be displayed
		if i > 0 and verbose:
			oi._debug("processed {} images".format((i + 1) * batchSize,
												   msgType="[PROGRESS]"))

	# finish the indexing process
	oi.finish()

	# dump the label encoder to file
	if verbose:
		print("[INFO] dumping labels to file...")
	labelEncoderPath = featuresPath[:featuresPath.rfind("/")] + "/le-" + fE[0] + ".cpickle"

	# confPath["label_encoder_path"][0:confPath["label_encoder_path"].rfind(".")] + "-" + fE[0] + ".cpickle"
	f = open(labelEncoderPath, "w")
	f.write(cPickle.dumps(le))
	f.close()

def generate_features(confPath, datasetPath, verbose):
	# shuffle the image paths to ensure randomness -- this will help make our
	# training and testing split code more efficient
	random.seed(42)
	random.shuffle(datasetPath)

	# determine the set of possible class labels from the image dataset assuming
	# that the images are in {directory}/{filename} structure and create the
	# label encoder
	if verbose:
		print("[INFO] encoding labels...")
	le = LabelEncoder()
	le.fit([p.split("/")[-2] for p in datasetPath])

	featureExtractors = confPath["featureExtractors"]
	outputPath = confPath["output_path"]
	datasetP = confPath["dataset_path"]
	batchSize = confPath["batch_size"]

	Parallel(n_jobs=-1)(delayed(extractFeatures)(fE, batchSize, datasetPath, outputPath,datasetP, le, verbose) for fE in featureExtractors)
	#for (fE) in featureExtractors:


def __main__():
	# construct the argument parser and parse the command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
	args = vars(ap.parse_args())

	# load the configuration and grab all image paths in the dataset
	conf = Conf(args["conf"])
	imagePaths = list(paths.list_images(conf["dataset_path"]))
	generate_features(conf,imagePaths)

if __name__ == "__main__":
    __main__()