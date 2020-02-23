# USAGE
# python index_features.py --conf conf/flowers17.json
# suppress any FutureWarning from Theano
from __future__ import print_function
from __future__ import absolute_import
import warnings

from imutils import paths

warnings.simplefilter(action="ignore", category=FutureWarning)
# import the necessary packages
from frimcla.extractor.extractor import Extractor
from frimcla.indexer.indexer import Indexer
from frimcla.utils.conf import Conf
from frimcla.utils import dataset

# from .extractor.extractor import Extractor
# from .indexer.indexer import Indexer
# from .utils.conf import Conf
# from .utils import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
# from guppy import hpy
import argparse
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import random
import os
import time
import re


# hp = hpy()

#This method list the dirs and files that are inside the path of the parameter
def ls(ruta = '.'):
    dir, subdirs, archivos = next(os.walk(ruta))
    return subdirs


"""
This method reduces the size of the parameter dataset until the user has the number of images chosen in the other parameter. 
The method randomly chooses the images that enter the dataset. The output is the balanced dataset.
"""
def cropDataset(datasetPath,n_images_dataset=100):
    subdirs = ls(datasetPath)
    n_classes = len(subdirs)
    images_path = []
    for subdir in subdirs:
        _, _, files = next(os.walk(os.path.join(datasetPath, subdir)))
        files = [os.path.join(datasetPath, subdir, file) for file in files]
        images_path+=files[0:int(n_images_dataset / n_classes)]
    return images_path

"""
	This method extract the features of the dataset and needs the feature extractor, the size of the batch, a list of 
	the images of the dataset, the path of the output, the path of the dataset, the label encoder (file that matchs the 
	clases with their respective encodings of the program) and if the user wants to read the information of the progress
	in the terminal.
"""
def extractFeatures(fE, batchSize, imagePaths, outputPath, datasetPath, le, verbose = False):
    imagePaths = [i.replace("\\ "," ")  for i in imagePaths]
    # initialize the Overfeat extractor and the Overfeat indexer
    if verbose:
        print("[INFO] initializing network...")
    featuresPath = outputPath + datasetPath[datasetPath.rfind("/"):] + \
				   "/models/features-" + fE[0] + ".hdf5"
    labelEncoderPath = featuresPath[:featuresPath.rfind("/")] + "/le.cpickle"
    directory = featuresPath[:featuresPath.rfind("/")]
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    f = open(labelEncoderPath, "wb")
    f.write(cPickle.dumps(le))
    f.close()
    if (not (os.path.isfile(featuresPath))):
        oe = Extractor(fE)
        oi = Indexer(featuresPath, estNumImages=len(imagePaths))
        if verbose:
            print("[INFO] starting feature extraction...")
        # loop over the image paths in batches
        for (i, paths) in enumerate(dataset.chunk(imagePaths, batchSize)):
            # load the set of images from disk and describe them
            (labels, images) = dataset.build_batch(paths, fE[0])
            features = oe.describe(images)
            # loop over each set of (label, vector) pair and add them to the indexer
            for (label, vector) in zip(labels, features):
                oi.add(label, vector)
                # check to see if progress should be displayed
            if i >= 0 and verbose:
                oi._debug("processed {} images".format((i + 1) * batchSize, msgType="[PROGRESS]"))
        # finish the indexing process
        oi.finish()
        # dump the label encoder to file
        if verbose:
            print("[INFO] dumping labels to file...")
    else:
        print("These features (" + fE[0] + ") are already generated")

"""
	This is the method that collects the features of all the feature extractors selected. The output are the features of each method that are stored in different files.
	Returns the execution time.

"""
def generateFeatures(outputPath, batchSize, datasetPath, featureExtractors, verbose=False,multiclass=False):
    start = time.time()
    # shuffle the image paths to ensure randomness -- this will help make our
    # training and testing split code more efficient
    imagePaths = cropDataset(datasetPath, 2000)
    # imagePaths = list(paths.list_images(datasetPath))
    random.seed(42)
    random.shuffle(imagePaths)
    # determine the set of possible class labels from the image dataset assuming
    # that the images are in {directory}/{filename} structure and create the
    # label encoder
    if verbose:
        print("[INFO] encoding labels...")

    if multiclass:
        le = MultiLabelBinarizer()
        le.fit([p.split(os.sep)[-2].split("_") for p in imagePaths])
    else:
        le = LabelEncoder()
        le.fit([p.split(os.sep)[-2] for p in imagePaths])
    # le.fit([p.split("/")[-2] for p in imagePaths])
    # Parallel(n_jobs=-1)(delayed(extractFeatures)(fE, batchSize, datasetPath, outputPath,datasetP, le, verbose) for fE in featureExtractors)
    for (fE) in featureExtractors:
    	# fParams.write(fE[0] + "," + fE[1])
    	extractFeatures(fE,batchSize, imagePaths, outputPath, datasetPath, le, verbose)
    finish = time.time()
    del le, imagePaths
    return finish-start

def __main__():
	# construct the argument parser and parse the command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
	args = vars(ap.parse_args())
	# load the configuration and label encoder
	conf = Conf(args["conf"])
	outputPath = conf["output_path"]
	datasetPath = conf["dataset_path"]
	featureExtractors = conf["feature_extractors"]
	batchSize = conf["batch_size"]
	# load the configuration and grab all image paths in the dataset
	# generateFeatures(conf,imagePaths,"False")
	generateFeatures(outputPath, batchSize, datasetPath, featureExtractors, False)

if __name__ == "__main__":
	__main__()