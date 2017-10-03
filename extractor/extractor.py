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

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

class Extractor:
	def __init__(self,modelText):
		# store the layer number and initialize the Overfeat transformer
		#self.layerNum = layerNum
		self.modelText=modelText
		print("[INFO] loading {}...".format(modelText))
		if modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
			Network = MODELS[modelText]
			self.model = Network(include_top=False)
		if modelText == "googlenet":
			self.model=GoogLeNetTransformer()
		if modelText == "overfeat":
			self.model = OverfeatTransformer(output_layers=[-3])
		if modelText == "lab888":
			self.model = LABModel()
		if modelText == "lab444":
			self.model = HSVModel(bins=[4,4,4])
		if modelText == "hsv888":
			self.model = LABModel()
		if modelText == "hsv444":
			self.model = HSVModel(bins=[4, 4, 4])
		if modelText == "haralick":
			self.model = Haralick()
		if modelText == "lbp":
			self.model = LBP()
		if modelText == "hog":
			self.model = HOG()
		if modelText == "haarhog":
			self.model = HaarHOG()
		if modelText == "densenet":
			self.model = DenseNet()
		if "annulus" in modelText:
			bags = int(modelText[modelText.find('_')+1:modelText.rfind('_')])
			p_segments = int(modelText[modelText.rfind('_')+1])
			self.model = HistogramsSeveralMasksAnnulusLabSegments(
				plainImagePath="/home/joheras/Escritorio/Research/Fungi/FungiImages/plain.jpg", bags=[bags,bags,bags],
				p_segments=p_segments)



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

