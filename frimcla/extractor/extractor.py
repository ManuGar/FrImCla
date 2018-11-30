# import the necessary packages
import numpy as np
from frimcla.shallowmodels.modelFactory import modelFactory
# from shallowmodels.modelFactory import modelFactory
import cv2


class Extractor:
	def __init__(self,modelText):
		# store the layer number and initialize the Overfeat transformer
		#self.layerNum = layerNum
		self.params =[]
		self.modelText=modelText[0]
		if len(modelText)>1:
			self.params= str(modelText[1])
		print("[INFO] loading {}...".format(modelText))

		modelFac = modelFactory()
		self.model = modelFac.getModel(self.modelText, self.params)

	def reshape(self,res):
		if(self.modelText=="resnet"):
			return np.reshape(res,1000)
		# return np.reshape(res, 2048)
		else:
			return res.flatten()

	def describe(self, images):
		if self.modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
			return [self.reshape(self.model.predict(image)) for image in images]
		if self.modelText in ("googlenet","overfeat"):
			return self.model.transform(images)
		# if self.modelText in ("mymodel"):
		# 	return self.model.describe(images)
		else:
			#print (self.modelText) #self.model.name
			return [self.model.describe(image) for image in images]