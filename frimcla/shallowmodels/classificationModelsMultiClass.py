from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLTSVM
from skmultilearn.adapt import MLkNN

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
        self.model = BinaryRelevance(RandomForestClassifier(random_state=random_state,n_estimators=n_estimators))
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
        self.model = BinaryRelevance(SVC(random_state=random_state))
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
        self.model = BinaryRelevance(KNeighborsClassifier())
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
        self.model = BinaryRelevance(LogisticRegression(random_state=rdm_state))
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
    def __init__(self, random_state=84,params={'classifier__activation':['identity', 'logistic', 'tanh', 'relu'],
                'classifier__solver':['lbfgs','sgd','adam'], 'classifier__alpha': sp_randint(0.0001, 1),
                'classifier__learning_rate':['constant','invscaling','adaptive'],'classifier__momentum':[0.9,0.95,0.99]},
                 niterations=5):
        self.model = BinaryRelevance(MLPClassifier(random_state=random_state))
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
                 params={"classifier__max_depth": [3, None],
				  "classifier__max_features": [1, 3, 10],
				  "classifier__min_samples_leaf": [1, 3, 10]},
                 niterations=10):
        self.model = BinaryRelevance(GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators))
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
    def __init__(self, random_state=84, n_estimators=20,params={'classifier__n_estimator': [250, 500, 1000, 1500],
                 'classifier__min_samples_split': [2, 4, 8]}, niterations=10):
        self.model = BinaryRelevance(ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators))
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



class ccRandomForest(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params = {"classifier__max_depth": [3, None],
				  "classifier__max_features": [1, 3, 10],
				  "classifier__min_samples_leaf": [1, 3, 10],
				  "classifier__bootstrap": [True, False],
				  "classifier__criterion": ["gini", "entropy"]}, niterations=10):
        self.model = ClassifierChain(RandomForestClassifier(random_state=random_state,n_estimators=n_estimators))
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

class ccSVM(classifierModel):
    def __init__(self, random_state=84, params={'classifier__C': [1, 10, 100, 1000],
                                                'classifier__gamma': [0.001, 0.0001],
                                                'classifier__kernel': ['rbf', 'linear']},
                 niterations=10):
        self.model = ClassifierChain(SVC(random_state=random_state))
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

class ccKNN(classifierModel):
    def __init__(self,params={'classifier__n_neighbors': range(5, 27,2)}, niterations=10):
        self.model = ClassifierChain(KNeighborsClassifier())
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

class ccLogRegression(classifierModel):
    def __init__(self, rdm_state=84,params={"classifier__C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
                 niterations=5):
        self.model = ClassifierChain(LogisticRegression(random_state=rdm_state))
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

class ccMultiLayerPerceptron(classifierModel):
    def __init__(self, random_state=84,params={'classifier__activation':['identity', 'logistic', 'tanh', 'relu'],
                'classifier__solver':['lbfgs','sgd','adam'], 'classifier__alpha': sp_randint(0.0001, 1),
                'classifier__learning_rate':['constant','invscaling','adaptive'],'classifier__momentum':[0.9,0.95,0.99]},
                 niterations=5):
        self.model = ClassifierChain(MLPClassifier(random_state=random_state))
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

class ccGradientBoost(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,
                 params={"classifier__max_depth": [3, None],
				  "classifier__max_features": [1, 3, 10],
				  "classifier__min_samples_leaf": [1, 3, 10]},
                 niterations=10):
        self.model = ClassifierChain(GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators))
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


class ccExtraTrees(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'classifier__n_estimators': [250, 500, 1000, 1500],
                 'classifier__min_samples_split': [2, 4, 8]}, niterations=10):
        self.model = ClassifierChain(ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators))
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


class lpRandomForest(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params = {"classifier__max_depth": [3, None],
				  "classifier__max_features": [1, 3, 10],
				  "classifier__min_samples_leaf": [1, 3, 10],
				  "classifier__bootstrap": [True, False],
				  "classifier__criterion": ["gini", "entropy"]}, niterations=10):
        self.model = LabelPowerset(RandomForestClassifier(random_state=random_state,n_estimators=n_estimators))
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

class lpSVM(classifierModel):
    def __init__(self, random_state=84, params={'classifier__C': [1, 10, 100, 1000],
                                                'classifier__gamma': [0.001, 0.0001],
                                                'classifier__kernel': ['rbf', 'linear']},
                 niterations=10):
        self.model = LabelPowerset(SVC(random_state=random_state))
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

class lpKNN(classifierModel):
    def __init__(self,params={'classifier__n_neighbors': range(5, 27,2)}, niterations=10):
        self.model = LabelPowerset(KNeighborsClassifier())
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

class lpLogRegression(classifierModel):
    def __init__(self, rdm_state=84,params={"classifier__C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
                 niterations=5):
        self.model = LabelPowerset(LogisticRegression(random_state=rdm_state))
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

class lpMultiLayerPerceptron(classifierModel):
    def __init__(self, random_state=84,params={'classifier__activation':['identity', 'logistic', 'tanh', 'relu'],
                'classifier__solver':['lbfgs','sgd','adam'], 'classifier__alpha': sp_randint(0.0001, 1),
                'classifier__learning_rate':['constant','invscaling','adaptive'],'classifier__momentum':[0.9,0.95,0.99]},
                 niterations=5):
        self.model = LabelPowerset(MLPClassifier(random_state=random_state))
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

class lpGradientBoost(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,
                 params={"classifier__max_depth": [3, None],
				  "classifier__max_features": [1, 3, 10],
				  "classifier__min_samples_leaf": [1, 3, 10]},
                 niterations=10):
        self.model = LabelPowerset(GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators))
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


class lpExtraTrees(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'classifier__n_estimators': [250, 500, 1000, 1500],
                 'classifier__min_samples_split': [2, 4, 8]}, niterations=10):
        self.model = LabelPowerset(ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators))
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

class mMLkNN(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'k': range(5,27,2),
                 's': [0.5, 0.7, 1.0]}, niterations=10):
        self.model = MLkNN()
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

class mMLTSVM(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'c_k': [2**i for i in range(-7, 7, 2)]}, niterations=10):
        self.model = MLTSVM()
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

