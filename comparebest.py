from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis

KFoldComparisionPath = "/home/magarcd/Escritorio/FrImCla/FrImCla/results/melanoma/kfold-comparison-resnet.csv"
index=KFoldComparisionPath.rfind("/")
path=KFoldComparisionPath[:index]
model= KFoldComparisionPath[index+1:KFoldComparisionPath.rfind(".")]
file = path+"/StatisticalComparison"+model[model.rfind("_"):]+".txt"
statisticalAnalysis(KFoldComparisionPath, file)
