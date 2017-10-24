from sklearn_theano.feature_extraction.caffe.googlenet import GoogLeNetTransformer
from sklearn_theano.feature_extraction import OverfeatTransformer
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from shallowmodels.models import LABModel,HSVModel,Haralick,LBP,HOG,HistogramsSeveralMasksAnnulusLabSegments,HaarHOG,DenseNet


MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

class modelFactory():

    def __init__(self):
        pass

    def getModel(self,modelText,params):

        #modificar el metodo para que acepte los parametros de los mode
        if modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
            Network = MODELS[modelText]
            return Network(include_top=params[0])
        if modelText == "googlenet":
            return GoogLeNetTransformer()
        if modelText == "overfeat":
            return OverfeatTransformer(output_layers=params[0]) #[-3]
        if modelText == "lab888":
            return LABModel()
        if modelText == "lab444":
            if (len(params)>=3):
                bin1=params[0]
                bin2=params[1]
                bin3=params[2]
                return HSVModel(bins=[bin1, bin2, bin3])
        if modelText == "hsv888":
            return LABModel()
        if modelText == "hsv444":
            if (len(params)>=3):
                bin1=params[0]
                bin2=params[1]
                bin3=params[2]
                return HSVModel(bins=[bin1, bin2, bin3])
            #return HSVModel(bins=[4, 4, 4])
        if modelText == "haralick":
            return Haralick()
        if modelText == "lbp":
            return LBP()
        if modelText == "hog":
            return HOG()
        if modelText == "haarhog":
            return HaarHOG()
        if modelText == "densenet":
            return DenseNet()
        if "annulus" in modelText:
            if (len(params)>=3):
                bags=params[0]
                p_segments =params[1]
                plainImgPath=params[2]
            #bags = int(modelText[modelText.find('_') + 1:modelText.rfind('_')])
            #p_segments = int(modelText[modelText.rfind('_') + 1])
            return HistogramsSeveralMasksAnnulusLabSegments(
                bags=[bags, bags, bags],
                p_segments=p_segments,
                plainImagePath=plainImgPath ) #"/home/joheras/Escritorio/Research/Fungi/FungiImages/plain.jpg",

