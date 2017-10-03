# USAGE
# python index_features.py --conf conf/flowers17.json

# suppress any FutureWarning from Theano
from __future__ import print_function
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

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and grab all image paths in the dataset
conf = Conf(args["conf"])
imagePaths = list(paths.list_images(conf["dataset_path"]))

# shuffle the image paths to ensure randomness -- this will help make our
# training and testing split code more efficient
random.seed(42)
random.shuffle(imagePaths)

# determine the set of possible class labels from the image dataset assuming
# that the images are in {directory}/{filename} structure and create the
# label encoder
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([p.split("/")[-2] for p in imagePaths])

# initialize the Overfeat extractor and the Overfeat indexer
print("[INFO] initializing network...")
oe = Extractor(conf["model"])
featuresPath = conf["features_path"][0:conf["features_path"].rfind(".")] + "-"+ conf["model"] +".hdf5"
oi = Indexer(featuresPath, estNumImages=len(imagePaths))
print("[INFO] starting feature extraction...")

# loop over the image paths in batches
for (i, paths) in enumerate(dataset.chunk(imagePaths, conf["batch_size"])):
	# load the set of images from disk and describe them
	(labels, images) = dataset.build_batch(paths, conf["model"])
	features = oe.describe(images)


	# loop over each set of (label, vector) pair and add them to the indexer
	for (label, vector) in zip(labels, features):
		oi.add(label, vector)

	# check to see if progress should be displayed
	if i > 0:
		oi._debug("processed {} images".format((i + 1) * conf["batch_size"],
			msgType="[PROGRESS]"))

# finish the indexing process
oi.finish()

# dump the label encoder to file
print("[INFO] dumping labels to file...")
labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
f = open(labelEncoderPath, "w")
f.write(cPickle.dumps(le))
f.close()