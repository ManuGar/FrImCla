#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================


from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from Comparing import compare_methods_h5py
import argparse
from utils.conf import Conf
from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
from sklearn.ensemble import ExtraTreesClassifier
import sys


from shallowmodels.classificationModelFactory import classificationModelFactory

def statisticalComparison(conf):
    for model in conf["model"]:

        print(model)
        featuresPath = conf["features_path"][0:conf["features_path"].rfind(".")] + "-" + model[0] + ".hdf5" #conf["model"]
        # db = h5py.File(featuresPath)
        # labels = db["image_ids"]
        labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-" + \
                           model[0] + ".cpickle" #conf["model"]
        # le = cPickle.loads(open(labelEncoderPath).read())
        # labels = [le.transform([l.split(":")[0]])[0] for l in labels]
        # df1 = pd.DataFrame([np.append(x,y) for (x,y) in zip(db["features"],labels)])
        featuresCSVPath = conf["features_csv_path"][0:conf["features_csv_path"].rfind(".")] + "-" + \
                          model[0] + ".csv" #conf["model"]

        # df1.to_csv(featuresCSVPath)
        # Loading dataset
        dataset = featuresCSVPath
        # df = pd.read_csv(featuresCSVPath)
        # data = df.ix[:, :-1].values
        listAlgorithms = []
        listParams = []
        for classificationModel in conf["modelClassifier"]:
            model = classificationModelFactory.getClassificationModel(classificationModel)
            cMo = model.getModel()
            params = model.get_params()
            listAlgorithms.append(cMo)
            listParams.append(params)

        print("-------------------------------------------------")
        print("Statistical Analysis")
        print("-------------------------------------------------")

        #listAlgorithms = [clfRF, clfSVC, clfKNN, clfLR, clfMLP]  # ,clfET
        #listParams = [param_distRF, param_distSVC, param_distKNN, param_distLR, param_distMLP]  # ,param_distET
        listNames = ["RF", "SVM", "KNN", "LR", "MLP"]  # ,"ET"

        resultsAccuracy = compare_methods_h5py(featuresPath, labelEncoderPath, listAlgorithms, listParams, listNames,
                                               [10, 10, 10, 5, 10], normalization=False)  # ,10

        dfAccuracy = pd.DataFrame.from_dict(resultsAccuracy, orient='index')
        KFoldComparisionPathAccuracy = conf["kfold_comparison"][0:conf["kfold_comparison"].rfind(".")] + "-" + \
                                       model[0] + ".csv"
        dfAccuracy.to_csv(KFoldComparisionPathAccuracy)

        statisticalAnalysis(KFoldComparisionPathAccuracy)
        # sys.stdout = sys.__stdout__

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())
conf = Conf(args["conf"])
statisticalComparison(conf)
