import frimcla.shallowmodels.classificationModelsMultiClass as cM

CLASSIFICATIONMODELS = {
    "RandomForest": cM.RandomForest(),
    "SVM" : cM.SVM(),
    "KNN" : cM.KNN(),
    "LogisticRegression" : cM.LogRegression(),
    "MLP" : cM.MultiLayerPerceptron(),
    "GradientBoost" : cM.GradientBoost(),
    "ExtraTrees" : cM.ExtraTrees(),
    "ccRandomForest": cM.ccRandomForest(),
    "ccSVM" : cM.ccSVM(),
    "ccKNN" : cM.ccKNN(),
    "ccLogisticRegression" : cM.ccLogRegression(),
    "ccMLP" : cM.ccMultiLayerPerceptron(),
    "ccGradientBoost" : cM.ccGradientBoost(),
    "ccExtraTrees" : cM.ccExtraTrees(),
    "lpRandomForest": cM.lpRandomForest(),
    "lpSVM" : cM.lpSVM(),
    "lpKNN" : cM.lpKNN(),
    "lpLogisticRegression" : cM.lpLogRegression(),
    "lpMLP" : cM.lpMultiLayerPerceptron(),
    "lpGradientBoost" : cM.lpGradientBoost(),
    "lpExtraTrees" : cM.lpExtraTrees(),
    "MLTSVM": cM.mMLTSVM(),
    "MLkNN": cM.mMLkNN(),
}

def ListClassificationModels():
    listClassiModels = []
    for name  in CLASSIFICATIONMODELS:
        listClassiModels.append(name)
    return listClassiModels

class classificationModelMultiClassFactory():

    def __init__(self):
        pass

    def getClassificationModel(self,modelText):
        return CLASSIFICATIONMODELS[modelText]
        # if modelText == "RandomForest":
        #     return cM.RandomForest()
        # elif modelText == "SVM":
        #     return cM.SVM()
        # elif modelText == "KNN":
        #     return cM.KNN()
        # elif modelText == "LogisticRegression":
        #     return cM.LogRegression()
        # elif modelText == "MLP":
        #     return cM.MultiLayerPerceptron()
        # elif modelText == "GradientBoost":
        #     return  cM.GradientBoost()
        # elif modelText == "ExtraTrees":
        #     return  cM.ExtraTrees()
