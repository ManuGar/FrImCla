import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

from scipy.stats import shapiro,levene,ttest_ind,wilcoxon
from multiprocessing.pool import ThreadPool

from .stac.nonparametric_tests import quade_test,holm_test,friedman_test
from .stac.parametric_tests import anova_test,bonferroni_test
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

def multipleAlgorithmsNonParametric(algorithms,accuracies, file, fileResults, alpha=0.05, verbose=False):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    filePath = file.name
    featureExtractor = filePath[filePath.rfind("_") + 1:filePath.rfind(("."))]

    if len(algorithms) < 5:
        if verbose:
            print("----------------------------------------------------------")
            print("Applying Quade test")
            print("----------------------------------------------------------")
        file.write("----------------------------------------------------------\n")
        file.write("Applying Quade test\n")
        file.write("----------------------------------------------------------\n")
        (Fvalue, pvalue, rankings, pivots) = quade_test(*accuracies)
    else:
        if verbose:
            print("----------------------------------------------------------")
            print("Applying Friedman test")
            print("----------------------------------------------------------")
        file.write("----------------------------------------------------------\n")
        file.write("Applying Friedman test\n")
        file.write("----------------------------------------------------------\n")
        (Fvalue, pvalue, rankings, pivots) = friedman_test(*accuracies)
    if verbose:
        print("F-value: %f, p-value: %s" % (Fvalue, pvalue))
    file.write("F-value: %f, p-value: %s \n" % (Fvalue, pvalue))

    if (pvalue < alpha):
        r = {x: y for (x, y) in zip(algorithms, rankings)}
        sorted_ranking = sorted(r.items(), key=operator.itemgetter(1))
        sorted_ranking.reverse()
        (winner, _) = sorted_ranking[0]
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}
        (comparions, zvalues, pvalues, adjustedpvalues) = holm_test(pivots, winner)
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)
        if verbose:
            print("Null hypothesis is rejected; hence, models have different performance")
            print (tabulate(sorted_ranking, headers=['Technique', 'Ranking']))
            print("Winner model: %s" % winner)
            print("----------------------------------------------------------")
            print("Applying Holm p-value adjustment procedure and analysing effect size")
            print("----------------------------------------------------------")
            print(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value']))

        file.write("Null hypothesis is rejected; hence, models have different performance \n")
        file.write(tabulate(sorted_ranking, headers=['Technique', 'Ranking']) + "\n")
        file.write("Winner model: %s \n" % winner)
        file.write("----------------------------------------------------------\n")
        file.write("Applying Holm p-value adjustment procedure and analysing effect size\n")
        file.write("----------------------------------------------------------\n")
        file.write(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value'])+ "\n")
        fileResults.write(winner)

        for ele in algorithmsDataset[winner]:
            fileResults.write(",")
            fileResults.write(str(ele))
        fileResults.write("\n")
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
                if verbose:
                    print("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                    winner, np.mean(algorithmsDataset[winner]), np.std(algorithmsDataset[winner]), c[c.rfind(" ") + 1:],
                    np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]), np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
                file.write("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s) \n" % (
                    winner, np.mean(algorithmsDataset[winner]), np.std(algorithmsDataset[winner]), c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]), np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))

            else:
                if verbose:
                    print("There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                        winner, np.mean(algorithmsDataset[winner]), np.std(algorithmsDataset[winner]), c[c.rfind(" ") + 1:],
                    np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]), np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
                file.write("There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s) \n" % (
                    winner, np.mean(algorithmsDataset[winner]),
                np.std(algorithmsDataset[winner]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))

    else:
        means = np.mean(accuracies, axis=1)
        maximum = max(means)

        if verbose:
            print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models")
            print("----------------------------------------------------------")
            print("Analysing effect size")
            print("----------------------------------------------------------")
            print("We take the model with the best mean (%s, mean: %f) and compare it with the other models: " % (
                algorithms[means.tolist().index(maximum)], maximum))

        file.write("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models\n")
        file.write("----------------------------------------------------------\n")
        file.write("Analysing effect size\n")
        file.write("----------------------------------------------------------\n")
        file.write("We take the model with the best mean (%s, mean: %f) and compare it with the other models: \n" % (
            algorithms[means.tolist().index(maximum)], maximum))
         #f = fileResults   confPath["classifier_path"] + conf["modelClassifiers"]             +filePath[filePath.rfind("_"):filePath.rfind(("."))]


        classifier = (algorithms[means.tolist().index(maximum)])
        fStatistical = open(filePath[:filePath.rfind("/")] + "/kfold-comparison_" + featureExtractor + ".csv", "r")
        fStatistical.readline()

        for line in fStatistical:
            line = line.split(",")
            if (line[0] == classifier):
                fileResults.write(line[0])
                for it in line[1:]:
                    fileResults.write("," + it)
        fStatistical.close()

        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"
                if verbose:
                    print("Comparing effect size of %s and %s: Cohen's d=%s, %s" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
                file.write("Comparing effect size of %s and %s: Cohen's d=%s, %s \n" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    if verbose:
        print("Eta squared: %f (%s)" % (eta,effectsize))
    file.write("Eta squared: %f (%s) \n" % (eta,effectsize))


def multipleAlgorithmsParametric(algorithms,accuracies, file, fileResults, alpha=0.05, verbose=False):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    (Fvalue, pvalue, pivots) = anova_test(*accuracies)
    if verbose:
        print("----------------------------------------------------------")
        print("Applying ANOVA test")
        print("----------------------------------------------------------")
        print("F-value: %f, p-value: %s" % (Fvalue, pvalue))

    file.write("----------------------------------------------------------\n")
    file.write("Applying ANOVA test\n")
    file.write("----------------------------------------------------------\n")
    file.write("F-value: %f, p-value: %s \n" % (Fvalue, pvalue))
    if (pvalue < alpha):
        if verbose:
            print("Null hypothesis is rejected; hence, models have different performance")
            print("----------------------------------------------------------")
            print("Applying Bonferroni-Dunn post-hoc and analysing effect size")
            print("----------------------------------------------------------")
        file.write("Null hypothesis is rejected; hence, models have different performance\n")
        file.write("----------------------------------------------------------\n")
        file.write("Applying Bonferroni-Dunn post-hoc and analysing effect size\n")
        file.write("----------------------------------------------------------\n")
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}

        (comparions, zvalues, pvalues, adjustedpvalues) = bonferroni_test(pivots, len(accuracies[0]))
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)

        if verbose:
            print(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value']))
        file.write(tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value'])+"\n")

        for (c, p) in zip(comparions, adjustedpvalues):
            cohend = abs(cohen_d(algorithmsDataset[c[0:c.find(" ")]], algorithmsDataset[c[c.rfind(" ") + 1:]]))
            if (cohend <= 0.2):
                effectsize = "Small"
            elif (cohend <= 0.5):
                effectsize = "Medium"
            else:
                effectsize = "Large"
            if (p > alpha):
                if verbose:
                    print("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))

                file.write("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
                #print("There are not significant differences between: %s and %s (Cohen's d=%s, %s)" % (c[0:c.find(" ")],c[c.rfind(" ") + 1:],cohend,effectsize))

                # if(np.mean(algorithmsDataset[c[0:c.find(" ")]])>np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]])):
                #     fileResults.write(c[0:c.find(" ")])
                #     for ele in algorithmsDataset[c[0:c.find(" ")]]:
                #         fileResults.write(",")
                #         fileResults.write(str(ele))
                #     fileResults.write("\n")
                # else:
                #     fileResults.write(c[c.rfind(" ") + 1:])
                #     for ele in algorithmsDataset[c[c.rfind(" ") + 1:]]:
                #         fileResults.write(",")
                #         fileResults.write(str(ele))
                #     fileResults.write("\n")

            else:
                if verbose:
                    print(
                    "There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)" % (
                    c[0:c.find(" ")],
                    np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                    np.std(algorithmsDataset[c[0:c.find(" ")]]),
                    c[c.rfind(" ") + 1:],
                    np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                    np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))

                file.write(
                "There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))

                # if (np.mean(algorithmsDataset[c[0:c.find(" ")]]) > np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]])):
                #     fileResults.write(c[0:c.find(" ")])
                #     for ele in algorithmsDataset[c[0:c.find(" ")]]:
                #         fileResults.write(",")
                #         fileResults.write(str(ele))
                #     fileResults.write("\n")
                # else:
                #     fileResults.write(c[c.rfind(" ") + 1:])
                #     for ele in algorithmsDataset[c[c.rfind(" ") + 1:]]:
                #         fileResults.write(",")
                #         fileResults.write(str(ele))
                #     fileResults.write("\n")


    else:
        means = np.mean(accuracies, axis=1)
        maximum = max(means)
        if verbose:
            print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models")
            print("----------------------------------------------------------")
            print("Analysing effect size")
            print("----------------------------------------------------------")
            print("We take the model with the best mean (%s, mean: %f) and compare it with the other models: " % (algorithms[means.tolist().index(maximum)],maximum))

        file.write("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models\n")
        file.write("----------------------------------------------------------\n")
        file.write("Analysing effect size\n")
        file.write("----------------------------------------------------------\n")
        file.write("We take the model with the best mean (%s, mean: %f) and compare it with the other models: \n" % (algorithms[means.tolist().index(maximum)],maximum))

        # filePath = file.name
        # featureExtractor = filePath[filePath.rfind("_") + 1:filePath.rfind(("."))]
        # classifier = (algorithms[means.tolist().index(maximum)])
        # #f = fileResults  # confPath["classifier_path"] + conf["modelClassifiers"]             +filePath[filePath.rfind("_"):filePath.rfind(("."))]
        # # print(filePath[:filePath.rfind("/")] + "/kfold-comparison_" + featureExtractor + ".csv")
        # fStatistical = open(filePath[:filePath.rfind("/")] + "/kfold-comparison_" + featureExtractor + ".csv", "r")
        # fStatistical.readline()
        # for line in fStatistical:
        #     line = line.split(",")
        #     # print(line[0][1:len(line[0])-1])
        #     # print(""+classifier)
        #     # print(line[0] == str(classifier))
        #     if (line[0] == classifier):
        #         fileResults.write(line[0])
        #         for it in line[1:]:
        #             fileResults.write(",")
        #             fileResults.write(it)
        # fStatistical.close()

        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"

                if verbose:
                    print("Comparing effect size of %s and %s: Cohen's d=%s, %s" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
                file.write("Comparing effect size of %s and %s: Cohen's d=%s, %s\n" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))

    means = np.mean(accuracies, axis=1)
    maximum = max(means)
    if verbose:
        print("----------------------------------------------------------")
        print("We take the model with the best mean (%s, mean: %f) and compare it with the other models: " % (
            algorithms[means.tolist().index(maximum)], maximum))

    file.write("----------------------------------------------------------\n")
    file.write("We take the model with the best mean (%s, mean: %f) and compare it with the other models: \n" % (
        algorithms[means.tolist().index(maximum)], maximum))

    filePath = file.name
    featureExtractor = filePath[filePath.rfind("_") + 1:filePath.rfind(("."))]
    classifier = (algorithms[means.tolist().index(maximum)])
    fStatistical = open(filePath[:filePath.rfind("/")] + "/kfold-comparison_" + featureExtractor + ".csv", "r")
    fStatistical.readline()
    for line in fStatistical:
        line = line.split(",")
        if (line[0] == classifier):
            fileResults.write(line[0])
            for it in line[1:]:
                fileResults.write(",")
                fileResults.write(it)
    fStatistical.close()




    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    if verbose:
        print("Eta squared: %f (%s)" % (eta,effectsize))
    file.write("Eta squared: %f (%s)\n" % (eta,effectsize))

def twoAlgorithmsParametric(algorithms,accuracies,alpha,file, fileResults, verbose=False):
    (t,prob)=ttest_ind(accuracies[0], accuracies[1])
    if verbose:
        print("Students' t: t=%f, p=%f" % (t,prob))
    file.write("Students' t: t=%f, p=%f \n" % (t,prob))
    # file.student("Students' t: t=%f, p=%f \n" % (t,prob))
    if (prob > alpha):
        if verbose: print("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s" % (
            algorithms[0], algorithms[1]))
        file.write("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s \n" % (
            algorithms[0], algorithms[1]))

        if(np.mean(accuracies[0])>np.mean(accuracies[1])):
            fileResults.write(algorithms[0])
            for ele in accuracies[0]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
        else:
            fileResults.write(algorithms[1])
            for ele in accuracies[1]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
    else:
        if verbose:
            print("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
        file.write("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f \n)" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
        if (np.mean(accuracies[0]) > np.mean(accuracies[1])):
            fileResults.write(algorithms[0])
            for ele in accuracies[0]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
        else:
            fileResults.write(algorithms[1])
            for ele in accuracies[1]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
    if verbose:
        print("----------------------------------------------------------")
        print("Analysing effect size")
        print("----------------------------------------------------------")
    file.write("----------------------------------------------------------\n")
    file.write("Analysing effect size\n")
    file.write("----------------------------------------------------------\n")
    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    if (prob <= alpha):
        if verbose: print("Cohen's d=%s, %s" % (cohend, effectsize))
        file.write("Cohen's d=%s, %s \n" % (cohend, effectsize))


def twoAlgorithmsNonParametric(algorithms,accuracies,alpha,file, fileResults, verbose=False):
    (t,prob)=wilcoxon(accuracies[0], accuracies[1])
    if verbose:
        print("Wilconxon: t=%f, p=%f" % (t,prob))
    file.write("Wilconxon: t=%f, p=%f \n" % (t,prob))
    if (prob > alpha):
        if verbose:
            print(
            "Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s" % (
            algorithms[0], algorithms[1]))
        file.write(
        "Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s \n" % (
            algorithms[0], algorithms[1]))

        if (np.mean(accuracies[0]) > np.mean(accuracies[1])):
            fileResults.write(algorithms[0])
            for ele in accuracies[0]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
        else:
            fileResults.write(algorithms[1])
            for ele in accuracies[1]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")

    else:
        if verbose:
            print("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))

        file.write("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) \n" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))

        if(np.mean(accuracies[0])>np.mean(accuracies[1])):
            fileResults.write( algorithms[0])
            for ele in accuracies[0]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")
        else:
            fileResults.write(algorithms[1])
            for ele in accuracies[1]:
                fileResults.write(",")
                fileResults.write(str(ele))
            fileResults.write("\n")

    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    if verbose:
        print("----------------------------------------------------------")
        print("Analysing effect size")
        print("----------------------------------------------------------")
    file.write("----------------------------------------------------------\n")
    file.write("Analysing effect size\n")
    file.write("----------------------------------------------------------\n")
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"

    if (prob <= alpha):
        if verbose:
            print("Cohen's d=%s, %s" % (cohend, effectsize))
        file.write("Cohen's d=%s, %s \n" % (cohend, effectsize))

def meanStdReportAndPlot(algorithms,accuracies,file, verbose= False):
    if verbose:
        print("**********************************************************")
        print("Mean and std")
        print("**********************************************************")

    file.write("**********************************************************\n")
    file.write("Mean and std \n")
    file.write("**********************************************************\n")
    means = np.mean(accuracies, axis=1)
    stds = np.std(accuracies, axis=1)
    for (alg, mean, std) in zip(algorithms, means, stds):
        msg = "%s: %f (%f)" % (alg, mean, std)
        if verbose: print(msg)
        file.write(msg+"\n")
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(np.transpose(accuracies))
    ax.set_xticklabels(algorithms)

    plt.savefig(file.name[:file.name.rfind("/")] + "/meansSTD" + file.name[file.name.rfind("_"):file.name.rfind(".")] + ".png")

def checkParametricConditions(accuracies,alpha, file, verbose=False):
    (W, p) = shapiro(accuracies)
    if verbose:
        print("Checking independence ")
        print("Ok")
        print("Checking normality using Shapiro-Wilk's test for normality, alpha=0.05")
        print("W: %f, p:%f" % (W, p))

    file.write("Checking independence \n")
    file.write("Ok\n")
    independence = True
    file.write("Checking normality using Shapiro-Wilk's test for normality, alpha=0.05\n")
    file.write("W: %f, p:%f \n" % (W, p))
    if p < alpha:
        if verbose: print("The null hypothesis (normality) is rejected")
        file.write("The null hypothesis (normality) is rejected\n")
        normality = False
    else:
        if verbose: print("The null hypothesis (normality) is accepted")
        file.write("The null hypothesis (normality) is accepted\n")
        normality = True
    (W, p) = levene(*accuracies)
    if verbose:
        print("Checking heteroscedasticity using Levene's test, alpha=0.05")
        print("W: %f, p:%f" % (W, p))

    file.write("Checking heteroscedasticity using Levene's test, alpha=0.05\n")
    file.write("W: %f, p:%f \n" % (W, p))
    if p < alpha:
        if verbose: print("The null hypothesis (heteroscedasticity) is rejected")
        file.write("The null hypothesis (heteroscedasticity) is rejected\n")
        heteroscedasticity = False
    else:
        if verbose: print("The null hypothesis (heteroscedasticity) is accepted")
        file.write("The null hypothesis (heteroscedasticity) is accepted\n")
        heteroscedasticity = True
    parametric = independence and normality and heteroscedasticity
    return parametric

# This is the main method employed to compare a dataset where the cross validation
# process has been already carried out.
def statisticalAnalysis(dataset, filePath, fileResults ,alpha=0.05, verbose=False):
    #path=dataset[:index]
    with open(filePath,"w") as file:  #path+"/StatisticalComparison"+model[model.rfind("_"):]+".txt"
        df = pd.read_csv(dataset)
        algorithms = df.ix[0:,0].values
        accuracies = df.ix[0:, 1:].values

        if (len(algorithms)<2):
            if verbose:
                print("It is neccessary to compare at least two algorithms")
            file.write("It is neccessary to compare at least two algorithms\n")
            fileResults.write(algorithms[0])
            for ele in accuracies[0]:
                fileResults.write(",") #str(ele)
                fileResults.write(str(ele))
            fileResults.write("\n")
            return

        if verbose:
            print(dataset)
            print(algorithms)
            print("==========================================================")
            print("Report")
            print("==========================================================")

        file.write(dataset)
        #file.write(algorithms)
        file.write("\n==========================================================\n")
        file.write("Report\n")
        file.write("==========================================================\n")
        meanStdReportAndPlot(algorithms,accuracies,file, verbose)

        if verbose:
            print("**********************************************************")
            print("Statistical tests")
            print("**********************************************************")
            print("----------------------------------------------------------")
            print("Checking parametric conditions ")
            print("----------------------------------------------------------")

        file.write("**********************************************************\n")
        file.write("Statistical test\n")
        file.write("**********************************************************\n")
        file.write("----------------------------------------------------------\n")
        file.write("Cheking parametric conditions\n")
        file.write("----------------------------------------------------------\n")
        parametric = checkParametricConditions(accuracies,alpha,file)

        if parametric:
            if verbose: print("Conditions for a parametric test are fulfilled")
            file.write("Conditions for a parametric test are fulfilled\n")
            if(len(algorithms)==2):
                if verbose:
                    print("----------------------------------------------------------")
                    print("Working with 2 algorithms")
                    print("----------------------------------------------------------")
                file.write("----------------------------------------------------------\n")
                file.write("Working with 2 algorithms\n")
                file.write("----------------------------------------------------------\n")
                twoAlgorithmsParametric(algorithms,accuracies,alpha,file, fileResults, verbose)
            else:
                if verbose:
                    print("----------------------------------------------------------")
                    print("Working with more than 2 algorithms")
                    print("----------------------------------------------------------")
                file.write("----------------------------------------------------------\n")
                file.write("Working with more than 2 algorithms\n")
                file.write("----------------------------------------------------------\n")
                multipleAlgorithmsParametric(algorithms,accuracies,file, fileResults, alpha, verbose)
        else:
            if verbose:
                print("Conditions for a parametric test are not fulfilled, applying a non-parametric test")
            file.write("Conditions for a parametric test are not fulfilled, applying a non-parametric test\n")
            if (len(algorithms) == 2):
                if verbose:
                    print("----------------------------------------------------------")
                    print("Working with 2 algorithms")
                    print("----------------------------------------------------------")
                file.write("----------------------------------------------------------\n")
                file.write("Working with 2 algorithms\n")
                file.write("----------------------------------------------------------\n")
                twoAlgorithmsNonParametric(algorithms, accuracies,alpha,file, fileResults, verbose)
            else:
                if verbose:
                    print("----------------------------------------------------------")
                    print("Working with more than 2 algorithms")
                    print("----------------------------------------------------------")
                file.write("----------------------------------------------------------\n")
                file.write("Working with more than 2 algorithms\n")
                file.write("----------------------------------------------------------\n")
                multipleAlgorithmsNonParametric(algorithms,accuracies,file, fileResults, alpha, verbose)
        file.close()

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