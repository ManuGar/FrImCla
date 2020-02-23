from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier


class classifierModel:
    def getModel(self):
        pass
    def getParams(self):
        pass
    def getNIterations(self):
        pass
    def setParams(self, params):
        pass
    def setNIterations(self,nIterations):
        pass

class RandomForest(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params = {"estimator__max_depth": [3, None],
				  "estimator__max_features": [1, 3, 10],
				  "estimator__min_samples_leaf": [1, 3, 10],
				  "estimator__bootstrap": [True, False],
				  "estimator__criterion": ["gini", "entropy"]}, niterations=10):
        self.model = OneVsRestClassifier(RandomForestClassifier(random_state=random_state,n_estimators=n_estimators))
        self.params = params
        self.niterations = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations

class SVM(classifierModel):
    def __init__(self, random_state=84, params={'estimator__C': [1, 10, 100, 1000],
                                                'estimator__gamma': [0.001, 0.0001],
                                                'estimator__kernel': ['rbf', 'linear']},
                 niterations=10):
        self.model = OneVsRestClassifier(SVC(random_state=random_state))
        self.params = params
        self.niterations= niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations

class KNN(classifierModel):
    def __init__(self,params={'estimator__n_neighbors': range(5, 27,2)}, niterations=10):
        self.model = OneVsRestClassifier(KNeighborsClassifier())
        self.params = params
        self.niteraciones = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niteraciones

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations

class LogRegression(classifierModel):
    def __init__(self, rdm_state=84,params={"estimator__C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
                 niterations=5):
        self.model = OneVsRestClassifier(LogisticRegression(random_state=rdm_state))
        self.params = params
        self.niterations = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations

class MultiLayerPerceptron(classifierModel):
    def __init__(self, random_state=84,params={'estimator__activation':['identity', 'logistic', 'tanh', 'relu'],
                'estimator__solver':['lbfgs','sgd','adam'], 'estimator__alpha': sp_randint(0.0001, 1),
                'estimator__learning_rate':['constant','invscaling','adaptive'],'estimator__momentum':[0.9,0.95,0.99]},
                 niterations=5):
        self.model = OneVsRestClassifier(MLPClassifier(random_state=random_state))
        self.params = params
        self.niterations = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations

class GradientBoost(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,
                 params={"estimator__max_depth": [3, None],
				  "estimator__max_features": [1, 3, 10],
				  "estimator__min_samples_leaf": [1, 3, 10]},
                 niterations=10):
        self.model = OneVsRestClassifier(GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators))
        self.params = params
        self.niterations = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations


class ExtraTrees(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'estimator__n_estimators': [250, 500, 1000, 1500],
                 'estimator__min_samples_split': [2, 4, 8]}, niterations=10):
        self.model = OneVsRestClassifier(ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators))
        self.params = params
        self.niterations = niterations

    def getModel(self):
        return self.model

    def getParams(self):
        return self.params

    def getNIterations(self):
        return self.niterations

    def setParams(self, params):
        self.params = params

    def setNIterations(self, nIterations):
        self.niterations = nIterations
