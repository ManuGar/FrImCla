# How to use the different files

There are different ways to use the framework, with jupyter notebooks, with a configuration file or using the API of the framework.
Jupyter notebooks are files that contain a guided process with embedded code on how to use FrImCla. In those notebooks the user can go running the code while we read what each algorithm is used for.
On the other hand we have configuration files. These files contain all the configuration that the framework needs to be executed. In this files, the user have to indicate parameters such as the feature extractor algorithms or the measure to use. This process is more automatic than the one mentioned before but it is also less interactive and not as guided.
Finally, using the API directly the users get completly freedom to use the algorithms as they want. This is the option for the most expert users.

Now we are going through the most important parts of the framework. 

## Performing a statistical comparison


1. Open the conf.json file and edit the variable dataset_path
   to indicate the path where the images are stored. Such a
   folder must have a subfolder for each class. You also have to indicate
   in output_path where the program's output will be saved.
2. In the conf.json file modify the variable feature_extractors to select
   the feature extractors and its params that you want to execute.
3. Extract the features of the dataset using the following
   command:
   python index_features.py -c conf/conf.json
   New methods can be added in the models.py file.
   Currently the methods that are available are inception, xception, vgg16, vgg19,
   resnet, googlenet, overfeat,lab888, lab444, hsv888, hsv444, haralick, lbp, hog,
   haarhog and densenet.
4. Open the conf.json file and edit the variable model_classifiers
   indicating the model/s that you want to use.
5. Perform the statistical analysis comparing different methods.
   New methods can be added in the classificationModels.py file.
   Currently the methods that are available are RandomForest, SVM, KNN, LR, MLP and
   RandomForest.
   To perform the analysis use the command:
   python StatisticalComparison.py -c conf/conf.json
   This will take some time but at the end you will get a report
   comparing the different methods.

## Training a classification model


Once that the user has determined the best model with the statistical
comparison, the user can create and train your model for further use.

1. Train the model executing the command:
   python train.py -c conf/conf.json

## Executing the full analysis


1. The best option to perform the full analysis of the methods (feature extractors and
   classifiers) is by executing the following comand:
   python fullAnalysis.py -c conf/conf.json

## Using the trained model to predict the class of a new image


After training the best model, you can use it to predict the class of
new images.

1. Execute the command:
   python prediction.py -c conf/conf.json -i imagePath
   In the above command you must replace imagePath with the path of the image.


## Adding new machine learning algorithms and feature extractor models


To add a new feature extractor models the user has to create a new class in models.py. This class implements Model and has to implement the method describe(self, image). The user also has to add this new model in modelFactory.py. In the method getModel(self,modelText) the user will create an object of the new class and its default params.

To add a new machine learning algorithm the user has to create a new class in classificationModels.py.
This class implements classifierModel.getModel and getters and setters for params and
nIterations. We also have to add this classifier model in classificationModelFactory.py. In the
method getClassificationModel(self,modelText) we will create an object of the new
class.
