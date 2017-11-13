#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================

import pandas as pd
from Comparing import compare_methods_h5py
import argparse
from utils.conf import Conf
from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
from sklearn.ensemble import ExtraTreesClassifier
import sys
import os


from shallowmodels.classificationModelFactory import classificationModelFactory

def statisticalComparison(conf):
    for model in conf["featureExtractors"]:

        print(model)

        featuresPath = conf["output_path"]+ conf["dataset_path"][conf["dataset_path"].rfind("/"):] + "/models/features-" + model[
			0] + ".hdf5"
        # db = h5py.File(featuresPath)
        # labels = db["image_ids"]
        labelEncoderPath = featuresPath[:featuresPath.rfind("/")]+ "/le-" + model[0] + ".cpickle"

        #conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-" + model[0] + ".cpickle"
        # le = cPickle.loads(open(labelEncoderPath).read())
        # labels = [le.transform([l.split(":")[0]])[0] for l in labels]
        # df1 = pd.DataFrame([np.append(x,y) for (x,y) in zip(db["features"],labels)])


        # df = pd.read_csv(featuresCSVPath)
        # data = df.ix[:, :-1].values
        factory =classificationModelFactory()
        listAlgorithms = []
        listParams = []
        listNiter = []

        filePathAux = conf["output_path"]+ conf["dataset_path"][conf["dataset_path"].rfind("/"):] + "/results/kfold-comparison_bestClassifiers.csv"
        filePath = conf["output_path"]+ conf["dataset_path"][conf["dataset_path"].rfind("/"):] + "/results/StatisticalComparison_" + model[0] + ".txt"
        if (not os.path.isfile(filePathAux)):
            fileResults = open(filePathAux, "a")
            fileResults.write(",0,1,2,3,4,5,6,7,8,9\n")
        else:
            fileResults = open(filePathAux, "a")

        for classificationModel in conf["modelClassifiers"]:
            print classificationModel
            modelClas = factory.getClassificationModel(classificationModel)
            cMo = modelClas.getModel()
            params = modelClas.getParams()
            niter = modelClas.getNIterations()
            listAlgorithms.append(cMo)
            listParams.append(params)
            listNiter.append(niter)

        listNames = conf["modelClassifiers"]

        print("-------------------------------------------------")
        print("Statistical Analysis")
        print("-------------------------------------------------")

        #listAlgorithms = [clfRF, clfSVC, clfKNN, clfLR, clfMLP]  # ,clfET
        #listParams = [param_distRF, param_distSVC, param_distKNN, param_distLR, param_distMLP]  # ,param_distET
        #listNames = ["RF", "SVM", "KNN", "LR", "MLP"]  # ,"ET"
        #Niteraciones de las clases [10, 10, 10, 5, 10]

        resultsAccuracy = compare_methods_h5py(featuresPath, labelEncoderPath, listAlgorithms, listParams, listNames,
                                               listNiter, normalization=False)  # ,10

        dfAccuracy = pd.DataFrame.from_dict(resultsAccuracy, orient='index')
        KFoldComparisionPathAccuracy =conf["output_path"]+ conf["dataset_path"][conf["dataset_path"].rfind("/"):] + "/results/kfold-comparison_"+model[0] + ".csv"
        #KFoldComparisionPathAccuracy=conf["kfold_comparison"][0:conf["kfold_comparison"].rfind(".")] + "-" + model[0] + ".csv"
        if (not os.path.exists(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])):
            os.mkdir(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])
        #"kfold_comparison": "results/minidatasetDogCat/kfold-comparison.csv"
        #"dataset_path": "/home/magarcd/Escritorio/ObjectClassificationByTransferLearning/ObjectClassificationByTransferLearning/minidatasetDogCat"
        dfAccuracy.to_csv(KFoldComparisionPathAccuracy)
        path = KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")]


        statisticalAnalysis(KFoldComparisionPathAccuracy,filePath, fileResults)
    fileResults.close()
    filePath2 = path + "/StatisticalComparison_bestClassifiers.txt"
    fileResults2 = open(path + "/" + "bestExtractorClassifier" + ".csv", "a")
    statisticalAnalysis(path + "/" + "kfold-comparison_bestClassifiers.csv", filePath2, fileResults2)
    fileResults2.close()
        # sys.stdout = sys.__stdout__

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())
conf = Conf(args["conf"])
statisticalComparison(conf)
