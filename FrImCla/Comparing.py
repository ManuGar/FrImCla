import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold
import h5py
import cPickle
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
    # tuple = [(i,(train_index,test_index),data,labels,listAlgorithms,listParameters,listAlgorithmNames,listNiters,normalization) for i,(train_index,test_index) in enumerate(kf.split(data))]
    # comparison = map(compare_method, tuple)
    #
    # for i,result in comparison:
    #     for name in listAlgorithmNames:
    #         results[name].append(result[name])

    for i,(train_index,test_index) in enumerate(kf.split(data)):
        if verbose:
            print "Iteration " + str(i+1) + "/10"

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


def accuracyPrediction(clf, params, name, n_iter, trainData, testData, trainLabels, testLabels, measure = "accuracy", verbose=False):
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
    # return (name, accuracy_score(testLabels, predictions))
    return (name, measure_score(testLabels,predictions, measure))


def compare_methods_h5py(model, featuresPath,labelEncoderPath,listAlgorithms,listParameters,listAlgorithmNames,
                         listNiters, measure, verbose=False, normalization=False):

    # Loading dataset
    db = h5py.File(featuresPath)
    labels = db["image_ids"]
    data = db["features"][()]
    fileAux = open(labelEncoderPath)
    le = cPickle.loads(fileAux.read())
    fileAux.close()
    labels = np.asarray([le.transform([l.split(":")[0]])[0] for l in labels])
    del le
    kf = KFold(n_splits=10,shuffle=False,random_state=42) #n_splits=10
    resultsAccuracy = {model[0]+ "_" +name:[] for name in listAlgorithmNames}

    for i,(train_index,test_index) in enumerate(kf.split(data)):
        if verbose:
            print "Iteration " + str(i)
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

        output = Parallel(n_jobs=-1)(delayed(accuracyPrediction)(clf, params, name, n_iter, trainData, testData, trainLabels, testLabels, measure,verbose)
                           for clf, params, name, n_iter in
                           zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters))

        #for clf, params, name, n_iter in zip(listAlgorithms, listParameters, listAlgorithmNames, listNiters):
        for name, accuracy in output:
            resultsAccuracy[model[0]+ "_" + name].append(accuracy)
        del output
    return resultsAccuracy


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