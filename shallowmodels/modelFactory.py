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

    def getModel(self,modelText):

        #modificar el metodo para que acepte los parametros de los mode
        if modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
            Network = MODELS[modelText]
            return Network(include_top=False)
        if modelText == "googlenet":
            return GoogLeNetTransformer()
        if modelText == "overfeat":
            return OverfeatTransformer(output_layers=[-3])
        if modelText == "lab888":
            return LABModel()
        if modelText == "lab444":
            return HSVModel(bins=[4, 4, 4])
        if modelText == "hsv888":
            return LABModel()
        if modelText == "hsv444":
            return HSVModel(bins=[4, 4, 4])
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
            bags = int(modelText[modelText.find('_') + 1:modelText.rfind('_')])
            p_segments = int(modelText[modelText.rfind('_') + 1])
            return HistogramsSeveralMasksAnnulusLabSegments(
                plainImagePath="/home/joheras/Escritorio/Research/Fungi/FungiImages/plain.jpg", bags=[bags, bags, bags],
                p_segments=p_segments)

