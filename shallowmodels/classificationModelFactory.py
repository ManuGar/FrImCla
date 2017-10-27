import classificationModels as cM

class classificationModelFactory():

    def __init__(self):
        pass

    def getClassificationModel(self,modelText):
        if modelText == "RandomForest":
            return cM.RandomForest()
        if modelText == "SVM":
            return cM.SVM()
        if modelText == "KNN":
            return cM.SVM()
        if modelText == "LogisticRegression":
            return cM.LogRegression()
        if modelText == "MultiLayerPerceptron":
            return cM.MultiLayerPerceptron()
        if modelText == "GradientBoost":
            return  cM.GradientBoost()
        if modelText == "ExtraTrees":
            return  cM.ExtraTrees()
