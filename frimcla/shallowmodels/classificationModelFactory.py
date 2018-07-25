import classificationModels as cM

class classificationModelFactory():

    def __init__(self):
        pass

    def getClassificationModel(self,modelText):
        if modelText == "RandomForest":
            return cM.RandomForest()
        elif modelText == "SVM":
            return cM.SVM()
        elif modelText == "KNN":
            return cM.KNN()
        elif modelText == "LogisticRegression":
            return cM.LogRegression()
        elif modelText == "MLP":
            return cM.MultiLayerPerceptron()
        elif modelText == "GradientBoost":
            return  cM.GradientBoost()
        elif modelText == "ExtraTrees":
            return  cM.ExtraTrees()
