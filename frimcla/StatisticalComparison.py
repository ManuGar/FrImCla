from __future__ import print_function
from __future__ import absolute_import
#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================

import argparse
import pandas as pd
import os
import json
import numpy as np
import h5py
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy import stats
# from .utils.conf import Conf
# from .Comparing import compare_methods_h5py, prepareModel
from frimcla.utils.conf import Conf
from frimcla.Comparing import compare_methods_h5py, prepareModel
from frimcla.StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
from frimcla.shallowmodels.classificationModelFactory import classificationModelFactory
from frimcla.shallowmodels.classificationModelMultiClassFactory import classificationModelMultiClassFactory
import time

#This list is used to say what combinations are not allowed
blacklist = [["haarhog", "SVM"],
             ["haarhog", "KNN"],
             ["haarhog","MLP"],
             ["haralick", "SVM"],
             ["haralick", "KNN"],
             ["hog", "SVM"],
             ["hog", "KNN"],
             ["hog", "LogisticRegression"]]
"""
    This method uses several models to predict the class of the images.
    The method returns the execution time.

"""
def majorityVoting(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, verbose= False):
    start = time.time()
    pathAux = outputPath + datasetPath[datasetPath.rfind("/"):]
    filePathAux = pathAux + "/results/combinationsAllowedMajorityVoting.txt"
    if not os.path.exists(pathAux + "/results"):
        os.makedirs(filePathAux[:filePathAux.rfind("/")])

    # predictions = []
    # modes = []
    # realimages = []
    combinations = []

    for fE in featureExtractors:
        if verbose:
            print(fE)

        featuresPath = pathAux + "/models/features-" + fE[0] + ".hdf5"
        labelEncoderPath = pathAux + "/models/le.cpickle"

        factory = classificationModelFactory()
        listAlgorithms = []
        listParams = []
        listNiter = []
        listNames = []
        fileAux = open(labelEncoderPath,"rb")
        le = cPickle.loads(fileAux.read())
        fileAux.close()
        fichero = open(filePathAux, "a")

        for classificationModel in modelClassifiers:
            combination = [fE[0], classificationModel]
            if (combination in blacklist):
                print("The combination(" + fE[0] + "-" + classificationModel + ") is not allowed")
            else:
                if verbose:
                    print(classificationModel)
                modelClas = factory.getClassificationModel(classificationModel)
                listAlgorithms.append(modelClas.getModel())
                listParams.append(modelClas.getParams())
                listNiter.append(modelClas.getNIterations())
                listNames.append(classificationModel)

        db = h5py.File(featuresPath)
        labels = db["image_ids"]
        data = db["features"][()]
        labels = np.asarray([le.transform([l.split(":")[0]])[0] for l in labels])
        kf = KFold(n_splits=10, shuffle=False, random_state=42)  # n_splits=10
        # __, (train_index, test_index) = enumerate(kf.split(data))
        # i, (train_index, test_index) = kf.split(data)
        (train_index, test_index) = next(kf.split(data), None)

        trainData, testData = data[train_index], data[test_index]
        trainLabels, testLabels = labels[train_index], labels[test_index]
        trainData = np.nan_to_num(trainData)
        testData = np.nan_to_num(testData)
        classif, models=prepareModel(trainData, trainLabels,testData, testLabels, listAlgorithms, listParams,
                                               listNames, listNiter, measure, verbose, normalization=False) # ,10)

        combinations.append( (fE[0], classif))
        # for mo in models:
            # prediction = mo.predict(testData)
            # prediction = le.inverse_transform(prediction)
            # fichero.write("Prediccion de " + str(fE[0]))
            # fichero.write(str(prediction) + "\n" )
            # predictions.append(prediction)


    # for j in testLabels:
    #     realimages.append(le.inverse_transform(j))


    '''
        With this code, the framework collects the mode of the columns of the matrix generated with the predictions.
    '''
    # aux = []
    # for i in range((len(testData))):
    #     for x in predictions:
    #         aux.append(x[i])
    #     aux = np.array(aux)
    #     mode = stats.mode(aux[0])
    #     # mode = le.inverse_transform(mode[0])
    #     # modes.append(le.inverse_transform(mode[0]))
    #     modes.append(mode[0])
    #     aux = []

    # measure = measure_score(testLabels, modes)
    # for mod in modes:
    #     fichero.write(str(le.inverse_transform(mod)) + "\n")
    # for ri in realimages:
    #     fichero.write(str(ri) + "\n")

    # fichero.write("El resultado de la medida es: " + str(measure))
    # fichero.close()

    fextractors = []
    cmodels = []
    for com in combinations:
        for fExtr in featureExtractors:
            if(fExtr[0]==com[0] and (len(com[1])>0)):
                classificationModels = com[1]
                fichero.write("For the feature extractor " + fExtr[0] + " there are available these classifiers: " + str(classificationModels) + "\n")
                if len(fExtr)==1:
                    fextractors.append({'model': fExtr[0], 'params': '', 'classificationModels':classificationModels})

                else:
                    fextractors.append({'model':  fExtr[0] , 'params': fExtr[1], 'classificationModels':classificationModels})

    for mod in modelClassifiers:
        cmodels.append(str(mod))

    fileConfModel = open(pathAux + "/ConfModel.json","w+")
    ConfModel={
          'featureExtractors': fextractors
          # 'classificationModel': cmodels
    }
    fichero.close()
    with fileConfModel as outfile:
        json.dump(ConfModel, outfile, indent=4)
    finish = time.time()
    return finish-start

"""
    This is the method of the second part of FrImCla. The input are the output path, dataset path, the list of feature 
    extractors that have been used in the previous step, the list of classification models, the measure that the user 
    wants to use and the verbose flag. The output is a list of files with the results of the statistical analysis and
    the combination of feature extractor and classifier model with the highest % of the measure selected by the user.
    The method returns the execution time.
"""
def statisticalComparison(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, nSteps=10, verbose= False,multiclass=False):
    start = time.time()
    pathAux = outputPath + datasetPath[datasetPath.rfind("/"):]
    filePathAux = pathAux + "/results/kfold-comparison_bestClassifiers.csv"
    if os.path.isfile(filePathAux):
        fileResults = open(filePathAux, "w")
    else:
        if not (os.path.exists(pathAux + "/results")):
            os.makedirs(filePathAux[:filePathAux.rfind("/")])
        fileResults = open(filePathAux, "w")
        for j in range(int(nSteps)):
            fileResults.write("," + str(j))
        fileResults.write("\n")

    alpha = 0.05
    for model in featureExtractors:
        if verbose:
            print(model)
        featuresPath = pathAux + "/models/features-" + model[0] + ".hdf5"
        labelEncoderPath = pathAux + "/models/le.cpickle"
        if multiclass:
            factory = classificationModelMultiClassFactory()
        else:
            factory =classificationModelFactory()
        listAlgorithms = []
        listParams = []
        listNiter = []
        listNames = []
        filePath = pathAux + "/results/StatisticalComparison_" + model[0] + ".txt"
        for classificationModel in modelClassifiers:
            combination = [model[0], classificationModel]
            if (combination in blacklist):
                print("The combination("+ model[0] + "-" + classificationModel + ") is not allowed")
            else:
                if verbose:
                    print(classificationModel)
                modelClas = factory.getClassificationModel(classificationModel)
                cMo = modelClas.getModel()
                params = modelClas.getParams()
                niter = modelClas.getNIterations()
                listAlgorithms.append(cMo)
                listParams.append(params)
                listNiter.append(niter)
                listNames.append(classificationModel)

        # if os.path.exists(pathAux + "/results"):
        #     if not os.path.isfile(filePathAux):
        #         fileResults = open(filePathAux, "a")
        #         for j in range(listNiter[0]):
        #             fileResults.write("," + str(j))
        #         fileResults.write("\n")
        #     else:
        #         fileResults = open(filePathAux, "a")
        # else:
        #     os.makedirs(filePathAux[:filePathAux.rfind("/")])
        #     fileResults = open(filePathAux, "a")
        #     for j in range(listNiter[0]):
        #         fileResults.write(","+str(j))
        #     fileResults.write("\n")
        if verbose:
            print("-------------------------------------------------")
            print("Statistical Analysis")
            print("-------------------------------------------------")
        #Niteraciones de las clases [10, 10, 10, 5, 10]
        resultsAccuracy = compare_methods_h5py(model, featuresPath, labelEncoderPath, listAlgorithms, listParams, listNames,
                                               listNiter,measure, nSteps, verbose, normalization=False,multiclass=multiclass)  # ,10

        dfAccuracy = pd.DataFrame.from_dict(resultsAccuracy, orient='index')
        KFoldComparisionPathAccuracy = pathAux + "/results/kfold-comparison_"+model[0] + ".csv"
        #KFoldComparisionPathAccuracy=conf["kfold_comparison"][0:conf["kfold_comparison"].rfind(".")] + "-" + model[0] + ".csv"
        if (not os.path.exists(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])):
            os.mkdir(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])
        dfAccuracy.to_csv(KFoldComparisionPathAccuracy)
        statisticalAnalysis(KFoldComparisionPathAccuracy,filePath, fileResults, alpha, verbose)
    fileResults.close()
    filePath2 = pathAux + "/results/StatisticalComparison_bestClassifiers.txt"
    fileResults2 = open(pathAux + "/results/bestExtractorClassifier.csv", "w")
    statisticalAnalysis(pathAux + "/results/kfold-comparison_bestClassifiers.csv", filePath2, fileResults2, alpha, verbose)
    fileResults2.close()
    file = open(pathAux + "/results/bestExtractorClassifier.csv")
    line = file.read()
    extractorClassifier = line.split(",")[0]
    extractor, classifier = extractorClassifier.split("_")

    for model in featureExtractors:
        if model[0]==extractor:
            if len(model)==1:
                parametros =""
            else:
                parametros=model[1]
            fileConfModel = open(pathAux + "/ConfModel.json","w+")
            if multiclass:
                ConfModel = {
                    'featureExtractors': [
                        {'model': model[0], 'params': str(parametros), 'classificationModels': [classifier],
                         'multiclass': True}],

                }
            else:
                ConfModel={
                    'featureExtractors': [{'model': model[0], 'params': str(parametros),'classificationModels': [classifier],
                                           'multiclass': False}],

                }
            with fileConfModel as outfile:
                json.dump(ConfModel, outfile, indent=4)
    del resultsAccuracy, dfAccuracy
    finish = time.time()
    return finish - start
        # sys.stdout = sys.__stdout__

def __main__():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    args = vars(ap.parse_args())
    # load the configuration and label encoder
    conf = Conf(args["conf"])
    outputPath = conf["output_path"]
    datasetPath = conf["dataset_path"]
    featureExtractors = conf["feature_extractors"]
    modelClassifiers = conf["model_classifiers"]
    measure = conf["measure"]
    statisticalComparison(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, False)

if __name__ == "__main__":
    __main__()