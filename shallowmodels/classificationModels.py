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

class RandomForest(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params = {"max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"]}):
        self.model = RandomForestClassifier(random_state=random_state,n_estimators=n_estimators)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class SVM(classifierModel):
    def __init__(self, random_state=84, params={'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf'], 'class_weight': ['balanced', None]}):
        self.model = SVC(random_state=random_state)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class KNN(classifierModel):
    def __init__(self,params={'n_neighbors': sp_randint(3, 30)}):
        self.model = KNeighborsClassifier()
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class LogisticRegression(classifierModel):
    def __init__(self, random_state=84,params={'C': [0.1, 0.5, 1, 10, 100, 1000]}):
        self.model = LogisticRegression(random_state=random_state)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class MultiLayerPerceptron(classifierModel):
    def __init__(self, random_state=84,params={'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                 'alpha': sp_randint(0.0001, 1), 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                 'momentum': [0.9, 0.95, 0.99]}):
        self.model = MLPClassifier(random_state=random_state)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class GradientBoost(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,
                 params={"max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "criterion": ["friedman_mse", "mse", "mae"]}):
        self.model = GradientBoostingClassifier(random_state=random_state,n_estimators=n_estimators)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params

class ExtraTrees(classifierModel):
    def __init__(self, random_state=84, n_estimators=20,params={'n_estimators': [250, 500, 1000, 1500],
                 'min_samples_split': [2, 4, 8]}):
        self.model = ExtraTreesClassifier(random_state=random_state,n_estimators=n_estimators)
        self.params = params
    def getModel(self):
        return self.model
    def getParams(self):
        return self.params