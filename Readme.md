# FrImCla
FrImCla is an open-source framework for Image Classification using traditional and deep learning tchniques. It supports a wide variety of deep learning and classical computer vision models. 

## Features
*   Several deep learning models
*   Classical computer vision techniques
*   Different performance measures
*   Jupyter notebook to use the framework
*   Ability to test several models at the same time
*   It is possible to easily add models and apply them into the framework to comprobe the performance over the dataset

## Requeriments and installation of the library
The library uses Python 2.7, which must be installed. 
The following packages must be installed:


*   numpy
*   scipy
*   scikit_learn
*   scikit-image
*   keras
*   h5py
*   OpenCV (i.e. cv2 must be available in python)
*   commentjson

## Documentation
Aquí poner el enlace al readme de como usarlo y demás

## Examples
Aquí poner los diferentes enlaces a los cuadernos

## List of feature extractor models

| Model | Description | Parameters | 
| --- | ------- | --- | --- |
| VGG16| Keras model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition. It has been obtained by directly converting the Caffe model provived by the authors. | *include_top*(either True or False). |
| VGG19 | Keras model of the 19-layer network used by the VGG team in the ILSVRC-2014 competition. It has been obtained by directly converting the Caffe model provived by the authors.| *include_top*(either True or False).  |
| Resnet |Residual neural networks utilizes skip connections or short-cuts to jump over some layers. In its limit as ResNets it will only skip over a single layer. With an additional weight matrix to learn the skip weights it is referred to as HighwayNets. With several parallel skips it is referred to as DenseNets. | *include_top*(either True or False).  |
| Inception | The network used a CNN inspired by LeNet but implemented a novel element which is dubbed an inception module. It used batch normalization and image distortions. This module is based on several very small convolutions in order to drastically reduce the number of parameters. | *include_top*(either True or False).
| Googlenet |  This network used a new variant of convolutional neural network called “Inception” for classification, and for detection the R-CNN was used. Google’s team was able to train a much smaller neural network and obtained much better results  compared to results obtained with convolutional neural networks in the previous year’s challenges. | No parameters. |
| Overfeat | It is a combination of CNN and another machine learning classifier ( LR, SVM, etc.). | *output_layers*(by default [-3]).  |
| Xception | This architecture has 36 convolutional layers forming the feature extraction base of the network. The Xception architecture is a linear stack of depthwise separable convolution layers with residual connections.| *include_top*(either True or False). 
| Densenet | This is a stack of dense blocks followed by transition layers. Each block consists of a series of units. Each unit packs two convolutions, each preceded by Batch Normalization and ReLU activations. Besides, each unit outputs a fixed number of feature vectors. This parameter, described as the growth rate, controls how much new information the layers allow to pass through. | No parameters |
| LAB888 | Histogram of the images with the lab colour format with 8 bins| No parameters. |
| LAB444 | Histogram of the images with the lab colour format with 4 bins | *bins*(by default 4,4,4). |
| HSV888 | Histogram of the images with the hsv colour format with 8 bins | No parameters. |
| HSV444 | Histogram of the images with the hsv colour format with 4 bins |*bins*(by default 4,4,4). | 
| Haralick | This model suggested the use of gray level co-occurrence matrices (GLCM). This method is based on the joint probability distributions of pairs of pixels. GLCM show how often each gray level occurs at a pixel located at a fixed geometric position relative to each other pixel, as a function of the gray level. An essential component is the definition of eight nearest-neighbor resolution cells that define different matrices for different angles (0°,45°,90°,135°) and distances between the horizontal neighboring pixels. | No parameters. | 
| Hog | The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy. | No parameters. |


## List of classifier models
*   Multilayer perceptron (MLP)
*   Support vector machine (SVM)
*   KNN  https://www.techopedia.com/definition/32066/k-nearest-neighbor-k-nn 
*   Logistic Regression
*   Random Forest


| Classifier | Description | Parameters | 
| --- | --- | --- | --- |
| Multilayer perceptron (MLP)| | |
| Support vector machine (SVM)| | |
| K-Nearest Neighbor(KNN) | | |
| Logistic Regression (LR)| | |
| Random Forest (RF)| | |


| Technique | Description | Parameters |
| --- | --- | --- |
| Multilayer perceptron (MLP)| | |
| Support vector machine (SVM) | | |
| K-Nearest Neighbor(KNN)| | |
| Logistic Regression (LR) | | |
| Random Forest (RF)| | |
