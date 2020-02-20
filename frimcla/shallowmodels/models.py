import cv2
import mahotas
# from skimage import exposure
from skimage import feature
# from imutils import auto_canny
import numpy as np
import math
import wget
from scipy.spatial import distance
from ..DenseNet import densenet
# from DenseNet import densenet
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.callbacks import LearningRateScheduler
from keras import optimizers

import zipfile
import os

class MyModel(Model):
    def __init__(self):
        # /home/magarcd/Escritorio/frimcla/frimcla/shallowmodels
        #./frimcla/shallowmodels/modeloRaices.h5
        # https://drive.google.com/file/d/1BLl9B4dryCbAvNPFb9LsNmoO4bqiSCaR/view?usp=sharing

        # https://www.dropbox.com/s/gnd9rbm0igogqrd/modeloRaices.h5?dl=0
        url = "https://www.dropbox.com/s/gnd9rbm0igogqrd/modeloRaices.h5?dl=1"
        file = wget.download(url, "./frimcla/shallowmodels/modeloRaices.h5")
        my_model = load_model(file)
        pruebaModel = Model(my_model.input, my_model.layers[-3].output)
        self.model = pruebaModel

    def describe(self,image):
        image = np.reshape(image,(64,64,1))
        image=np.expand_dims(image, axis=0)
        return self.model.predict(image)

class Histonet(Model):
    def __init__(self):
        my_model = load_model("frimcla/shallowmodels/histonet.h5")
        pruebaModel = Model(my_model.input, my_model.layers[-1].output)
        self.model = pruebaModel

    def describe(self,image):
        return self.model.predict(image)


class DenseNet(Model):
    def __init__(self):
        modelI = densenet.DenseNet(depth=40, growth_rate=12, bottleneck=True, reduction=0.5)
        modelI.layers.pop()
        modelI.layers.pop()
        modelI.outputs = [modelI.layers[-1].output]
        modelI.layers[-1].outbound_nodes = []

        new_input = modelI.input
        hidden_layer = modelI.layers[-2].output
        new_output = Flatten()(hidden_layer)
        super(DenseNet,self).__init__()
        self.model = Model(new_input, new_output)

    def describe(self,image):

        '''
        Lo de que se tenga qeu poner solo la componente 0 de ese vector revisar porque se ha tenido que poner para que funcione la prediccion
        no deberia ser asi. En el entrenamiento ha valido sin eso
        '''
        return self.model.predict(image)[0]

class LABModel(Model):
    def __init__(self,bins=[8,8,8],channels=[0,1,2],histValues=[0,256,0,256,0,256]):
        self.bins =bins
        self.channels=channels
        self.histValues = histValues

    def describe(self,image):
        checkLAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        dst = np.zeros(shape=(5, 2))

        histLAB = cv2.calcHist([checkLAB], self.channels, None, self.bins, self.histValues)
        histLAB = cv2.normalize(histLAB, dst).flatten()
        return histLAB

    # def describe(self, image):
    #     checkLAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #     histLAB = cv2.calcHist([checkLAB], self.channels, None, self.bins, self.histValues)
    #     histLAB = cv2.normalize(histLAB).flatten()
    #     return histLAB


class HSVModel(Model):
    def __init__(self,bins=[8,8,8],channels=[0,1,2],histValues=[0,180,0,256,0,256]):
        self.bins =bins
        self.channels=channels
        self.histValues = histValues

    def describe(self,image):
        checkHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        dst = np.zeros(shape=(5, 2))
        histHSV = cv2.calcHist([checkHSV], self.channels, None, self.bins, self.histValues)
        histHSV = cv2.normalize(histHSV, dst).flatten()
        return histHSV
    # def describe(self,image):
    #     checkHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     histHSV = cv2.calcHist([checkHSV], self.channels, None, self.bins, self.histValues)
    #     histHSV = cv2.normalize(histHSV).flatten()
    #     return histHSV

class Haralick(Model):
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = mahotas.features.haralick(gray).mean(axis=0)
        return features

class LBP(Model):
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = mahotas.features.lbp(gray, 3, 24)
        return features

class HOG(Model):
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True)
        return features


class HaarHOG(Model):
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        featuresHOG = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True)
        featuresHaar = mahotas.features.haralick(gray).mean(axis=0)
        return np.append(featuresHOG,featuresHaar)

class HistogramsSeveralMasksAnnulusLabSegments(Model):

    def __init__(self,plainImagePath,bags=[8,8,8],channels=[0,1,2],histValues=[0,256,0,256,0,256],p_segments=2):
        self.plainImagePath = plainImagePath
        self.bags = bags
        self.channels = channels
        self.histValues=histValues
        self.p_segments=p_segments

    def describe(self,image):
        (h,w) = image.shape[:2]
        control = image[0:h,0:w/2]
        control = cv2.resize(control, (100, 100))
        plain = cv2.imread(self.plainImagePath)
        plain = cv2.resize(plain, (100, 100))
        check = image[0:h,w/2:w]
        check = cv2.resize(check, (100, 100))
        combinations = [(control * float(n) / 100 + plain * float(100 - n) / 100).astype("uint8") for n in
                        range(1, 101, 1)]
        combinationPercentage = [((100 - n)) for n in range(1, 101, 1)]

        segments = 2**self.p_segments
        # Mask to only keep the centre
        mask = np.zeros(control.shape[:2], dtype="uint8")

        (h, w) = control.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        masks = [mask.copy() for i in range(0, 8 * segments)]
        # Generating the different annulus masks
        for i in range(0, 8 * segments):
            cv2.circle(masks[i], (cX, cY), min(90 - 10 * (i % 8), control.shape[1]) / 2, 255, -1)
            cv2.circle(masks[i], (cX, cY), min(80 - 10 * (i % 8), control.shape[1]) / 2, 0, -1)

        if (self.p_segments == 2):
            points = np.array([[cX, cY], [cX, 0], [0, 0], [0, h], [w, h], [w, cY], [cX, cY]], np.int32)
            points = points.reshape((-1, 1, 2))
            for i in range(0, 8):
                cv2.fillConvexPoly(masks[i], points, 0)
        else:
            for k in range(0, 2 ** (self.p_segments - 2)):
                alpha = (math.pi / 2 ** (self.p_segments - 1)) * (k + 1)
                beta = (math.pi / 2 ** (self.p_segments - 1)) * k
                if alpha <= math.pi / 4:
                    points = np.array([[cX, cY], [w, h / 2 - w / 2 * math.tan(alpha)], [w, 0], [0, 0], [0, h], [w, h],
                                       [w, h / 2 - w / 2 * math.tan(beta)], [cX, cY]], np.int32)
                    points = points.reshape((-1, 1, 2))
                    points2 = np.array([[cX, cY], [w, cY], [w, h / 2 - w / 2 * math.tan(beta)], [cX, cY]], np.int32)
                    points2 = points2.reshape((-1, 1, 2))
                    for i in range(0, 8):
                        cv2.fillConvexPoly(masks[8 * k + i], points, 0)
                        cv2.fillConvexPoly(masks[8 * k + i], points2, 0)


                else:
                    points = np.array([[cX, cY], [cX + (h / 2) / math.tan(alpha), 0], [0, 0], [0, h], [w, h], [w, 0],
                                       [cX + (h / 2) / math.tan(beta), 0], [cX, cY]], np.int32)
                    points = points.reshape((-1, 1, 2))
                    points2 = np.array([[cX, cY], [cX + (h / 2) / math.tan(beta), 0], [w, 0], [w, cY], [cX, cY]],
                                       np.int32)
                    points2 = points2.reshape((-1, 1, 2))
                    for i in range(0, 8):
                        cv2.fillConvexPoly(masks[8 * k + i], points, 0)
                        cv2.fillConvexPoly(masks[8 * k + i], points2, 0)

        M90 = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
        M180 = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
        M270 = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)

        for i in range(0, 8 * (2 ** (self.p_segments - 2))):
            masks[8 * (2 ** (self.p_segments - 2)) + i] = cv2.warpAffine(masks[i], M90, (w, h))
            masks[2 * 8 * (2 ** (self.p_segments - 2)) + i] = cv2.warpAffine(masks[i], M180, (w, h))
            masks[3 * 8 * (2 ** (self.p_segments - 2)) + i] = cv2.warpAffine(masks[i], M270, (w, h))
        results = []


        for mask in masks:

            checkLAB = cv2.cvtColor(check, cv2.COLOR_RGB2LAB)

            histLAB = cv2.calcHist([checkLAB], self.channels, mask, self.bags, self.histValues)
            histLAB = cv2.normalize(histLAB).flatten()
            histsLAB = [cv2.normalize(
                cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_RGB2LAB)],
                             self.channels, mask, self.bags, self.histValues)).flatten() for im in combinations]
            # Compare histograms
            comparisonLABeuclidean = [distance.euclidean(histLAB, histLAB2) for histLAB2 in histsLAB]
            mins = np.where(np.asarray(comparisonLABeuclidean) == np.asarray(comparisonLABeuclidean).min())
            results.append([[combinationPercentage[n], comparisonLABeuclidean[n]] for n in mins[0].tolist()])

        percentageNew = []
        for p in results:
            if p[0][0] > 60:
                percentageNew.append(p[np.argmax(p, axis=0)[0]])
            else:
                percentageNew.append(p[np.argmin(p, axis=0)[0]])

        percentage = [p[0] for p in percentageNew]
        return (percentage)