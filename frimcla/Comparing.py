from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold
import h5py
import re
import os
import _pickle as cPickle
# import cPickle
from sklearn.externals.joblib import Parallel, delayed

def apply_algorithm(tuple):
    clf, params, name, n_iter,trainData, trainLabels,testData,testLabels = tuple
    print(name)
    if params is None:
        model = clf
    else:
        model = RandomizedSearchCV(clf, param_distributions=params, n_iter=n_iter)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    return (name,accuracy_score(testLabels, predictions))

def compare_method(tuple):
    i, (train_index, test_index),data,labels,listAlgorithms,listParameters,listAlgorithmNames,listNiters,normalization = tuple
    print("Iteration " + str(i+1))
    results = {name: [] for name in listAlgorithmNames}
    trainData, testData = data[train_index], data[test_index]
    trainLabels, testLabels = labels[train_index], labels[test_index]

    tuple = [(clf, params, name, n_iter, trainData, trainLabels, testData, testLabels) for clf, params, name, n_iter in
             zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters)]
    #p = ThreadPool(len(listAlgorithms))
    comparison = map(apply_algorithm, tuple)
    for (name, comp) in comparison:
        results[name].append(comp)
    print("Finished iteration " + str(i))
    return (i,results)

def compare_methods(dataset,listAlgorithms,listParameters,listAlgorithmNames,listNiters,normalization=False, verbose=False):

    # Loading dataset
    df = pd.read_csv(dataset)
    data = df.ix[:, :-1].values
    labels = df.ix[:, -1].values
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    resultsAccuracy = {name:[] for name in listAlgorithmNames}
    #resultsAUROC = {name: [] for name in listAlgorithmNames}
    resultsPrecision = {name: [] for name in listAlgorithmNames}
    resultsRecall = {name: [] for name in listAlgorithmNames}
    resultsFmeasure = {name: [] for name in listAlgorithmNames}

    for i,(train_index,test_index) in enumerate(kf.split(data)):
        if verbose:
            print("Iteration " + str(i+1) + "/10")

        trainData , testData = data[train_index],data[test_index]
        trainLabels, testLabels = labels[train_index], labels[test_index]

        # Normalization
        if normalization:
            trainData = np.asarray(trainData).astype("float64")
            trainData -= np.mean(trainData, axis=0)
            trainData /= np.std(trainData, axis=0)
            testData = np.asarray(testData).astype("float64")
            testData -= np.mean(testData, axis=0)
            testData /= np.std(testData, axis=0)
        # tuple = [(clf,params,name,n_iter,trainData, trainLabels,testData,testLabels) for clf,params,name,n_iter in zip(listAlgorithms,listParameters,listAlgorithmNames,listNiters)]
        # p = Pool(len(listAlgorithms))
        # comparison = p.map(apply_algorithm, tuple)
        # for (name, comp) in comparison:
        #     results[name].append(comp)
        #
        for clf,params,name,n_iter in zip(listAlgorithms,listParameters,listAlgorithmNames,listNiters):
            if verbose:
                print(name)
            if params is None:
                model = clf
            else:
                model = RandomizedSearchCV(clf, param_distributions=params,n_iter=n_iter)
            model.fit(trainData, trainLabels)
            predictions = model.predict(testData)
            resultsAccuracy[name].append(accuracy_score(testLabels, predictions))
            #resultsAUROC[name].append(roc_auc_score(testLabels, predictions))
            resultsPrecision[name].append(precision_score(testLabels, predictions))
            resultsRecall[name].append(recall_score(testLabels, predictions))
            resultsFmeasure[name].append(f1_score(testLabels, predictions))

    return (resultsAccuracy,resultsPrecision,resultsRecall,resultsFmeasure)


def trainModel(clf, params, name, n_iter, trainData, trainLabels, verbose=False):
    if verbose:
        print(name)
    if params is None:
        model = clf
    else:
        model = RandomizedSearchCV(clf, param_distributions=params, n_iter=n_iter)
    model.fit(trainData, trainLabels)
    outp = (name, model)
    return outp


def prepareModel(trainData, trainLabels,testData, testLabels, listAlgorithms, listParameters,
                                               listAlgorithmNames, listNiters, measure, verbose=False, normalization=False):
    combinations = []
    output = []
    # Normalization
    if normalization:
        trainData = np.asarray(trainData).astype("float32")
        trainData -= np.mean(trainData, axis=0)
        trainData /= np.std(trainData, axis=0)
        testData = np.asarray(testData).astype("float32")
        testData -= np.mean(testData, axis=0)
        testData /= np.std(testData, axis=0)

    output = Parallel(n_jobs=-1)(delayed(trainModel)(clf, params, name, n_iter, trainData, trainLabels, verbose)
                        for clf, params, name, n_iter in
                        zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters))
    models = []
    for (name, model) in output:
        predictions = model.predict(testData)
        if measure_score(predictions, testLabels, measure) > 0.56:
            models.append(model)
            combinations.append(name)
    del output
    return (combinations,models)


"""
    This method is used to predict the class of the images and compare with the test labels to know how good works 
    the model. Returns the name of the model and the result of the comparison of the predictions with the test labels 
    using the measure selected by the user. 
"""
def measurePrediction(clf, params, name, n_iter, trainData, testData, trainLabels, testLabels, measure = "accuracy", verbose=False):
    if verbose:
        print(name)
    if params is None:
        model = clf
    else:
        model = RandomizedSearchCV(clf, param_distributions=params, n_iter=n_iter)

    trainData = np.nan_to_num(trainData)
    testData = np.nan_to_num(testData)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    return (name, measure_score(testLabels,predictions, measure))

"""
    This method performs a statistical analysis of the methods for a feature extractor. The output is a list of the 
    classification models with their results using the measure selected by the user. 
"""
def compare_methods_h5py(model, featuresPath,labelEncoderPath,listAlgorithms,listParameters,listAlgorithmNames,
                         listNiters, measure, nSteps, verbose=False, normalization=False,multiclass=False):
    # Loading dataset
    db = h5py.File(featuresPath)
    labels = db["image_ids"]
    data = db["features"][()]
    fileAux = open(labelEncoderPath, "rb")
    le = cPickle.loads(fileAux.read())
    fileAux.close()
    if multiclass:
        labels = np.array([list(le.transform([re.split(":|\\\\", l)[-2].split('_')])[0]) for l in labels])
    else:
        labels = np.asarray([le.transform([re.split(":|\\\\" , l)[-2]])[0] for l in labels])
    # labels = np.asarray([le.transform([l.split(":")[0]])[0] for l in labels])
    del le
    kf = KFold(n_splits=int(nSteps), shuffle=False,random_state=42) #n_splits=10
    resultsAccuracy = {model[0]+ "_" +name:[] for name in listAlgorithmNames}

    for i,(train_index,test_index) in enumerate(kf.split(data)):
        if verbose:
            print("Iteration " + str(i))
        trainData , testData = data[train_index],data[test_index]
        trainLabels, testLabels = labels[train_index], labels[test_index]

        # Normalization
        if normalization:
            trainData = np.asarray(trainData).astype("float32")
            trainData -= np.mean(trainData, axis=0)
            trainData /= np.std(trainData, axis=0)
            testData = np.asarray(testData).astype("float32")
            testData -= np.mean(testData, axis=0)
            testData /= np.std(testData, axis=0)
        output = Parallel(n_jobs=-1)(delayed(measurePrediction)(clf, params, name, n_iter, trainData, testData, trainLabels, testLabels, measure,verbose)
                           for clf, params, name, n_iter in
                           zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters))

        #for clf, params, name, n_iter in zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters):
        for name, accuracy in output:
            resultsAccuracy[model[0]+ "_" + name].append(accuracy)
        del output
    return resultsAccuracy

"""
    With this method the user obtain the results of comparing the test labels with the prediction labels of a 
    classifier model. The measure is selected by the user.
"""
def measure_score(testLabels, predictions, measure_name = "accuracy"):
    if measure_name == "accuracy":
        return accuracy_score(testLabels, predictions)
    elif measure_name == "f1":
        return f1_score(testLabels, predictions)
    elif measure_name == "recall":
        return recall_score(testLabels, predictions)
    elif measure_name == "precision":
        return precision_score(testLabels, predictions)
    elif measure_name == "auroc":
        return roc_auc_score(testLabels, predictions)
    else:
        pass