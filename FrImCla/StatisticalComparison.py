#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================

import pandas as pd
from Comparing import compare_methods_h5py
import argparse
from utils.conf import Conf
from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
import os


from shallowmodels.classificationModelFactory import classificationModelFactory


blacklist = [["haarhog", "SVM"],
             ["haarhog", "KNN"],
             ["haralick", "SVM"],
             ["haralick", "KNN"],
             ["hog", "SVM"],
             ["hog", "KNN"]]

def statisticalComparison(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, verbose= False):
    pathAux = outputPath + datasetPath[datasetPath.rfind("/"):]
    filePathAux = pathAux + "/results/kfold-comparison_bestClassifiers.csv"

    for model in featureExtractors:

        if verbose:
            print(model)

        featuresPath = pathAux + "/models/features-" + model[0] + ".hdf5"
        # db = h5py.File(featuresPath)
        # labels = db["image_ids"]
        labelEncoderPath = pathAux + "/models/le-" + model[0] + ".cpickle"

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

        # listNames = modelClassifiers
        if verbose:
            print("-------------------------------------------------")
            print("Statistical Analysis")
            print("-------------------------------------------------")

        #listAlgorithms = [clfRF, clfSVC, clfKNN, clfLR, clfMLP]  # ,clfET
        #listParams = [param_distRF, param_distSVC, param_distKNN, param_distLR, param_distMLP]  # ,param_distET
        #listNames = ["RF", "SVM", "KNN", "LR", "MLP"]  # ,"ET"
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
    fileResults2 = open(pathAux + "/results/bestExtractorClassifier" + ".csv", "a")
    statisticalAnalysis(pathAux + "/results/kfold-comparison_bestClassifiers.csv", filePath2, fileResults2, verbose)
    fileResults2.close()
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


    # statisticalComparison(conf)
    statisticalComparison(outputPath, datasetPath, featureExtractors, modelClassifiers, measure, False)

if __name__ == "__main__":
    __main__()