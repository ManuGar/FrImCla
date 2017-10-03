import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.stats import shapiro,levene,ttest_ind,wilcoxon
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from stac.nonparametric_tests import quade_test,holm_test,friedman_test
from stac.parametric_tests import anova_test,bonferroni_test
from tabulate import tabulate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score,f1_score
from sklearn.model_selection import KFold

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def SSbetween(accuracies):
    return float(sum(accuracies.sum(axis=1)**2))/len(accuracies[0]) - float(accuracies.sum()**2)/(len(accuracies[0])*len(accuracies))

def SSTotal(accuracies):
    sum_y_squared = sum([value**2 for value in accuracies.flatten()])
    return sum_y_squared - float(accuracies.sum() ** 2) / (len(accuracies[0]) * len(accuracies))

def eta_sqrd(accuracies):
    return SSbetween(accuracies)/SSTotal(accuracies)

def multipleAlgorithmsNonParametric(algorithms,accuracies,alpha=0.05):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    if len(algorithms) < 5:
        print("----------------------------------------------------------")
        print("Applying Quade test")
        print("----------------------------------------------------------")
        (Fvalue, pvalue, rankings, pivots) = quade_test(*accuracies)
    else:
        print("----------------------------------------------------------")
        print("Applying Friedman test")
        print("----------------------------------------------------------")
        (Fvalue, pvalue, rankings, pivots) = friedman_test(*accuracies)
    print("F-value: %f, p-value: %s" % (Fvalue, pvalue))
    if (pvalue < alpha):
        print("Null hypothesis is rejected; hence, models have different performance")
        r = {x: y for (x, y) in zip(algorithms, rankings)}
        sorted_ranking = sorted(r.items(), key=operator.itemgetter(1))
        sorted_ranking.reverse()
        print tabulate(sorted_ranking, headers=['Technique', 'Ranking'])
        (winner, _) = sorted_ranking[0]
        print("Winner model: %s" % winner)
        print("----------------------------------------------------------")
        print("Applying Holm p-value adjustment procedure and analysing effect size")
        print("----------------------------------------------------------")
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}

        (comparions, zvalues, pvalues, adjustedpvalues) = holm_test(pivots, winner)
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)

        print(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value']))
        for (c, p) in zip(comparions, adjustedpvalues):
            cohend = abs(cohen_d(algorithmsDataset[winner], algorithmsDataset[c[c.rfind(" ") + 1:]]))
            if (cohend <= 0.2):
                effectsize = "Small"
            elif (cohend <= 0.5):
                effectsize = "Medium"
            else:
                effectsize = "Large"
            if (p > alpha):
                #print("There are not significant differences between: %s and %s (Cohen's d=%s, %s)" % (
                #winner, c[c.rfind(" ") + 1:], cohend, effectsize))
                print("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                    winner, np.mean(algorithmsDataset[winner]),
                np.std(algorithmsDataset[winner]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
            else:
                print("There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                    winner, np.mean(algorithmsDataset[winner]),
                np.std(algorithmsDataset[winner]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
    else:
        print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models")
        print("----------------------------------------------------------")
        print("Analysing effect size")
        print("----------------------------------------------------------")
        means = np.mean(accuracies, axis=1)

        maximum = max(means)
        print("We take the model with the best mean (%s, mean: %f) and compare it with the other models: " % (
        algorithms[means.tolist().index(maximum)], maximum))
        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"

                print("Comparing effect size of %s and %s: Cohen's d=%s, %s" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    print("Eta squared: %f (%s)" % (eta,effectsize))


def multipleAlgorithmsParametric(algorithms,accuracies,alpha=0.05):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    print("----------------------------------------------------------")
    print("Applying ANOVA test")
    print("----------------------------------------------------------")
    (Fvalue, pvalue, pivots) = anova_test(*accuracies)
    print("F-value: %f, p-value: %s" % (Fvalue, pvalue))
    if (pvalue < alpha):
        print("Null hypothesis is rejected; hence, models have different performance")
        print("----------------------------------------------------------")
        print("Applying Bonferroni-Dunn post-hoc and analysing effect size")
        print("----------------------------------------------------------")
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}

        (comparions, zvalues, pvalues, adjustedpvalues) = bonferroni_test(pivots, len(accuracies[0]))
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)

        print(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value']))

        for (c, p) in zip(comparions, adjustedpvalues):
            cohend = abs(cohen_d(algorithmsDataset[c[0:c.find(" ")]], algorithmsDataset[c[c.rfind(" ") + 1:]]))
            if (cohend <= 0.2):
                effectsize = "Small"
            elif (cohend <= 0.5):
                effectsize = "Medium"
            else:
                effectsize = "Large"
            if (p > alpha):
                print("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
                #print("There are not significant differences between: %s and %s (Cohen's d=%s, %s)" % (c[0:c.find(" ")],c[c.rfind(" ") + 1:],cohend,effectsize))
            else:
                print(
                "There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
    else:
        print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models")
        print("----------------------------------------------------------")
        print("Analysing effect size")
        print("----------------------------------------------------------")
        means = np.mean(accuracies, axis=1)
        maximum = max(means)
        print("We take the model with the best mean (%s, mean: %f) and compare it with the other models: " % (algorithms[means.tolist().index(maximum)],maximum))
        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"

                print("Comparing effect size of %s and %s: Cohen's d=%s, %s" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    print("Eta squared: %f (%s)" % (eta,effectsize))




def twoAlgorithmsParametric(algorithms,accuracies,alpha):
    (t,prob)=ttest_ind(accuracies[0], accuracies[1])
    print("Students' t: t=%f, p=%f" % (t,prob))
    if (prob > alpha):
        print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s" % (
            algorithms[0], algorithms[1]))
    else:
        print("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
    print("----------------------------------------------------------")
    print("Analysing effect size")
    print("----------------------------------------------------------")
    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    if (prob <= alpha):
        print("Cohen's d=%s, %s" % (cohend, effectsize))


def twoAlgorithmsNonParametric(algorithms,accuracies,alpha):
    (t,prob)=wilcoxon(accuracies[0], accuracies[1])
    print("Wilconxon: t=%f, p=%f" % (t,prob))
    if (prob > alpha):
        print(
        "Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s" % (
            algorithms[0], algorithms[1]))
    else:
        print("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    print("----------------------------------------------------------")
    print("Analysing effect size")
    print("----------------------------------------------------------")
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"

    if (prob <= alpha):
        print("Cohen's d=%s, %s" % (cohend, effectsize))





def meanStdReportAndPlot(algorithms,accuracies,dataset):
    print("**********************************************************")
    print("Mean and std")
    print("**********************************************************")
    means = np.mean(accuracies, axis=1)
    stds = np.std(accuracies, axis=1)
    for (alg, mean, std) in zip(algorithms, means, stds):
        msg = "%s: %f (%f)" % (alg, mean, std)
        print(msg)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(np.transpose(accuracies))
    ax.set_xticklabels(algorithms)
    plt.savefig("meansSTD.png")

def checkParametricConditions(accuracies,alpha):
    print("Checking independence ")
    print("Ok")
    independence = True
    print("Checking normality using Shapiro-Wilk's test for normality, alpha=0.05")
    (W, p) = shapiro(accuracies)
    print("W: %f, p:%f" % (W, p))
    if p < alpha:
        print("The null hypothesis (normality) is rejected")
        normality = False
    else:
        print("The null hypothesis (normality) is accepted")
        normality = True
    print("Checking heteroscedasticity using Levene's test, alpha=0.05")
    (W, p) = levene(*accuracies)
    print("W: %f, p:%f" % (W, p))
    if p < alpha:
        print("The null hypothesis (heteroscedasticity) is rejected")
        heteroscedasticity = False
    else:
        print("The null hypothesis (heteroscedasticity) is accepted")
        heteroscedasticity = True

    parametric = independence and normality and heteroscedasticity
    return parametric


# This is the main method employed to compare a dataset where the cross validation
# process has been already carried out.
def statisticalAnalysis(dataset,alpha=0.05):
    df = pd.read_csv(dataset)
    algorithms = df.ix[0:,0].values
    if (len(algorithms)<2):
        print("It is neccessary to compare at least two algorithms")
        return
    accuracies = df.ix[0:,1:].values
    print(dataset)
    print(algorithms)
    print("==========================================================")
    print("Report")
    print("==========================================================")
    meanStdReportAndPlot(algorithms,accuracies,dataset)
    print("**********************************************************")
    print("Statistical tests")
    print("**********************************************************")
    print("----------------------------------------------------------")
    print("Checking parametric conditions ")
    print("----------------------------------------------------------")
    parametric = checkParametricConditions(accuracies,alpha)

    if parametric:
        print("Conditions for a parametric test are fulfilled")
        if(len(algorithms)==2):
            print("----------------------------------------------------------")
            print("Working with 2 algorithms")
            print("----------------------------------------------------------")
            twoAlgorithmsParametric(algorithms,accuracies,alpha)
        else:
            print("----------------------------------------------------------")
            print("Working with more than 2 algorithms")
            print("----------------------------------------------------------")
            multipleAlgorithmsParametric(algorithms,accuracies,alpha)
    else:
        print("Conditions for a parametric test are not fulfilled, applying a non-parametric test")
        if (len(algorithms) == 2):
            print("----------------------------------------------------------")
            print("Working with 2 algorithms")
            print("----------------------------------------------------------")
            twoAlgorithmsNonParametric(algorithms, accuracies,alpha)
        else:
            print("----------------------------------------------------------")
            print("Working with more than 2 algorithms")
            print("----------------------------------------------------------")
            multipleAlgorithmsNonParametric(algorithms, accuracies,alpha)

def compare_method(tuple):
    iteration, train_index,test_index,data,labels, clf, params, name,metric = tuple
    trainData, testData = data[train_index], data[test_index]
    trainLabels, testLabels = labels[train_index], labels[test_index]
    #print("Iteration " + str(iteration) + " of " + name)
    if params is None:
        model = clf
        model.fit(trainData, trainLabels)
    else:
        try:
            model = RandomizedSearchCV(clf, param_distributions=params, n_iter=20,random_state=84)
            model.fit(trainData, trainLabels)
        except ValueError:
            model = RandomizedSearchCV(clf, param_distributions=params, n_iter=5,random_state=84)
            model.fit(trainData, trainLabels)

    predictions = model.predict(testData)
    try:
        if (metric == 'accuracy'):
            return (name, accuracy_score(testLabels, predictions))
        elif (metric == 'recall'):
            return (name, recall_score(testLabels, predictions))
        elif (metric == 'precision'):
            return (name, precision_score(testLabels, predictions))
        elif (metric == 'f1'):
            return (name, f1_score(testLabels, predictions))
        elif (metric == 'auroc'):
            return (name, roc_auc_score(testLabels, predictions))
    except ValueError:
        print("In the multiclass problem, only accuracy can be used as metric")
        return



def compare_methods(data,labels,listAlgorithms,listParameters,listAlgorithmNames,metric='accuracy',alpha=0.5):

    if(metric!='accuracy' and metric!='recall' and metric!='precision' and metric!='f1' and metric!='auroc'):
        print("Invalid metric")
        return


    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    results = {name:[] for name in listAlgorithmNames}


    tuple = [(i,train_index,test_index,data,labels,x,y,z,metric) for i,(train_index,test_index) in enumerate(kf.split(data))
                  for (x, y, z) in zip(listAlgorithms, listParameters, listAlgorithmNames)]

    p = ThreadPool(len(listAlgorithms))

    comparison=p.map(compare_method,tuple)
    for (name,comp) in comparison:
        results[name].append(comp)

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('temp.csv')
    statisticalComparison('temp.csv',alpha=alpha)

