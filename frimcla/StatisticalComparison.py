#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================

import os
import json
import argparse
import pandas as pd
from utils.conf import Conf
from Comparing import compare_methods_h5py
from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
from shallowmodels.classificationModelFactory import classificationModelFactory

#This list is used to say what combinations are not allowed
blacklist = [["haarhog", "SVM"],
             ["haarhog", "KNN"],
             ["haralick", "SVM"],
             ["haralick", "KNN"],
             ["hog", "SVM"],
             ["hog", "KNN"],
             ["hog", "LogisticRegression"]]

"""
    This is the method of the second part of FrImCla. The input are the output path, dataset path, the list of feature 
    extractors that have been used in the previous step, the list of classification models, the measure that the user 
    wants to use and the verbose flag. The output is a list of files with the results of the statistical analysis and
    the combination of feature extractor and classifier model with the highest % of the measure selected by the user.
"""
def statisticalComparison(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, verbose= False):
    pathAux = outputPath + datasetPath[datasetPath.rfind("/"):]
    filePathAux = pathAux + "/results/kfold-comparison_bestClassifiers.csv"

    for model in featureExtractors:
        if verbose:
            print(model)
        featuresPath = pathAux + "/models/features-" + model[0] + ".hdf5"
        labelEncoderPath = pathAux + "/models/le-" + model[0] + ".cpickle"
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
                    print classificationModel
                modelClas = factory.getClassificationModel(classificationModel)
                cMo = modelClas.getModel()
                params = modelClas.getParams()
                niter = modelClas.getNIterations()
                listAlgorithms.append(cMo)
                listParams.append(params)
                listNiter.append(niter)
                listNames.append(classificationModel)
        if os.path.exists(pathAux + "/results"):
            if not os.path.isfile(filePathAux):
                fileResults = open(filePathAux, "a")
                for j in range(listNiter[0]):
                    fileResults.write("," + str(j))
                fileResults.write("\n")
            else:
                fileResults = open(filePathAux, "a")
        else:
            os.makedirs(filePathAux[:filePathAux.rfind("/")])
            fileResults = open(filePathAux, "a")
            for j in range(listNiter[0]):
                fileResults.write(","+str(j))
            fileResults.write("\n")
        if verbose:
            print("-------------------------------------------------")
            print("Statistical Analysis")
            print("-------------------------------------------------")
        #Niteraciones de las clases [10, 10, 10, 5, 10]
        resultsAccuracy = compare_methods_h5py(model, featuresPath, labelEncoderPath, listAlgorithms, listParams, listNames,
                                               listNiter,measure , verbose, normalization=False)  # ,10
        dfAccuracy = pd.DataFrame.from_dict(resultsAccuracy, orient='index')
        KFoldComparisionPathAccuracy = pathAux + "/results/kfold-comparison_"+model[0] + ".csv"
        #KFoldComparisionPathAccuracy=conf["kfold_comparison"][0:conf["kfold_comparison"].rfind(".")] + "-" + model[0] + ".csv"
        if (not os.path.exists(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])):
            os.mkdir(KFoldComparisionPathAccuracy[:KFoldComparisionPathAccuracy.rfind("/")])
        dfAccuracy.to_csv(KFoldComparisionPathAccuracy)
        statisticalAnalysis(KFoldComparisionPathAccuracy,filePath, fileResults,verbose)
    fileResults.close()
    filePath2 = pathAux + "/results/StatisticalComparison_bestClassifiers.txt"
    fileResults2 = open(pathAux + "/results/bestExtractorClassifier.csv", "a")
    statisticalAnalysis(pathAux + "/results/kfold-comparison_bestClassifiers.csv", filePath2, fileResults2, verbose)
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
            ConfModel={
                    'featureExtractor': {'model': model[0], 'params': str(parametros)},
                    'classificationModel': classifier
                }
            with fileConfModel as outfile:
                json.dump(ConfModel, outfile)
    del resultsAccuracy, dfAccuracy
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