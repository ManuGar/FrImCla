from ast import literal_eval

# from sklearn_theano.feature_extraction.caffe.googlenet import GoogLeNetTransformer
# from sklearn_theano.feature_extraction import OverfeatTransformer
from ..sklearn_theano.feature_extraction.caffe.googlenet import GoogLeNetTransformer
from ..sklearn_theano.feature_extraction import OverfeatTransformer
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from .models import LABModel,HSVModel,Haralick,LBP,HOG,HistogramsSeveralMasksAnnulusLabSegments,HaarHOG,DenseNet, MyModel


MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50,
    "googlenet": GoogLeNetTransformer,
    "overfeat": OverfeatTransformer,
    "lab888": LABModel,
    "lab444": LABModel,
    "hsv888": HSVModel,
    "hsv444": HSVModel,
    "haralick": Haralick,
    "lbp": LBP,
    "hog": HOG,
    "haarhog": HaarHOG,
    "densenet": DenseNet,
    "annulus": HistogramsSeveralMasksAnnulusLabSegments,
    "mymodel": MyModel
}

def listFeatureExtractors():
    listModels = []
    for name in MODELS:
        listModels.append(name)
    return listModels


class modelFactory():

    def __init__(self):
        pass

    def getModel(self,modelText,params):
        if modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
            if params=="True" or params =="False":
                Network = MODELS[modelText]
                return Network(include_top=params[0])
            else:
                raise ValueError('The number of parameters is not correct')
        elif modelText == "overfeat":
            params = str(params)
            params = literal_eval(params)
            if len(params)==1:
                return MODELS[modelText](output_layers=params) #"[-3]"
            else:
                raise ValueError('The number of parameters is not correct')
        elif modelText in ("hsv444","lab444" ):
            params = [int(s) for s in params.split(',')]
            if (len(params)==3):
                bin1=params[0]
                bin2=params[1]
                bin3=params[2]
                return MODELS[modelText](bins=[bin1, bin2, bin3])
            else:
                raise ValueError('The number of parameters is not correct')
        elif modelText in ("googlenet", "lab888", "hsv888", "haralick", "lbp", "hog", "haarhog", "densenet","mymodel"):
            return MODELS[modelText]()
        # elif "annulus" in modelText:
        #     params = [int(s) for s in params.split(',')]
        #     if (len(params)==3):
        #         bags=params[0]
        #         p_segments =params[1]
        #         plainImgPath=params[2]
        #     else: raise ValueError('The number of parameters is not correct')
        #     #bags = int(modelText[modelText.find('_') + 1:modelText.rfind('_')])
        #     #p_segments = int(modelText[modelText.rfind('_') + 1])
        #     return MODELS[modelText](
        #         bags=[bags, bags, bags],
        #         p_segments=p_segments,
        #         plainImagePath=plainImgPath ) #"/home/joheras/Escritorio/Research/Fungi/FungiImages/plain.jpg",
