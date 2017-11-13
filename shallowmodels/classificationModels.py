from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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
    def __init__(self, random_state=84, n_estimators=20,params = {"max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"]}, niterations=10):
        self.model = RandomForestClassifier(random_state=random_state,n_estimators=n_estimators)
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
    def __init__(self, random_state=84, params={'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf'], 'class_weight': ['balanced', None]}, niterations=10):
        self.model = SVC(random_state=random_state)
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
    def __init__(self,params={'n_neighbors': sp_randint(3, 30)}, niterations=10):
        self.model = KNeighborsClassifier()
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
    def __init__(self, rdm_state=84,params={'C': [0.1, 0.5, 1, 10, 100, 1000]}, niterations=5):
        self.model = LogisticRegression(random_state=rdm_state)
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
    def __init__(self, random_state=84,params={'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                 'alpha': sp_randint(0.0001, 1), 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                 'momentum': [0.9, 0.95, 0.99]}, niterations=10):
        self.model = MLPClassifier(random_state=random_state)
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
                 params={"max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "criterion": ["friedman_mse", "mse", "mae"]}, niterations=10):
        self.model = GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators)
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
    def __init__(self, random_state=84, n_estimators=20,params={'n_estimators': [250, 500, 1000, 1500],
                 'min_samples_split': [2, 4, 8]}, niterations=10):
        self.model = ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators)
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
