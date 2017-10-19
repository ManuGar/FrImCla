# import the necessary packages

from sklearn_theano.feature_extraction.caffe.googlenet import GoogLeNetTransformer
from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
from sklearn_theano.feature_extraction import OverfeatTransformer
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
import numpy as np
from shallowmodels.models import LABModel,HSVModel,Haralick,LBP,HOG,HistogramsSeveralMasksAnnulusLabSegments,HaarHOG,DenseNet
from shallowmodels.modelFactory import modelFactory

class Extractor:
	def __init__(self,modelText):
		# store the layer number and initialize the Overfeat transformer
		#self.layerNum = layerNum
		self.modelText=modelText
		print("[INFO] loading {}...".format(modelText))
		modelFac = modelFactory()
		self.model = modelFac.getModel(modelText)  #the model choice moved to modelFactory.py

	def reshape(self,res):
		if(self.modelText=="resnet"):
			return np.reshape(res,2048)
		else:
			return res.flatten()


	def describe(self, images):
		if self.modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
			return [self.reshape(self.model.predict(image)) for image in images]
		if self.modelText in ("googlenet","overfeat"):
			return self.model.transform(images)
		else:
			return [self.model.describe(image) for image in images]

