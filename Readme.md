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











































| Technique | Description | Parameters | Change annotation |
| --- | --- | --- | --- |
| Average blurring| Smoothes the image using an average filter. | *kernel*: Kernel size for average blurring (either 3, 5, 7, 9 or 11).| No |
| Bilateral blurring | Applies bilateral blurring to the image. | *diameter*: Diameter size for bilateral blurring (integer value). *sigmaColor*: sigma color for bilateral blurring (integer value). *sigmaSpace*: sigma space for bilateral blurring (integer value)  | No |
| Blurring | Blurs an image using the normalized box filter. | *ksize*: Kernel size for blurring (either 3, 5, 7, 9 or 11).  | No |
| Change to HSV | Changes the color space from RGB to HSV. | No parameters. | No |
| Change to LAB | Changes the color space from RGB to LAB. | No parameters.| No |
| Crop | Crops pixels at the sides of the image. | *percentage*:  Percentage to keep during cropping (value between 0 and 1). *startFrom*: Position to start the cropping ("TOPLEFT", "TOPRIGHT", "BOTTOMLEFT", "BOTTOMRIGHT", "CENTER","TOPLEFT")  | Yes |
| Dropout | Sets some pixels in the image to zero. | *percentage*: Percentage of pixels to drop (value between 0 and 1). | No |
| Elastic deformation | Applies elastic deformation as explained in the paper:  P. Simard, D. Steinkraus, and J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. Proceedings of the 12th International Conference on Document Analysis and Recognition (ICDAR'03) vol. 2, pp. 958--964. IEEE Computer Society. 2003. | *alpha*:  Alpha value for elastic deformation. *sigma*: Sigma value for elastic deformation | Yes |
| Equalize histogram | Applies histogram equalization to the image. | No parameters. | No |
| Flip | Flips the image horizontally, vertically or both. | *flip*: Flip value: 1 for vertical flip, 0 for horizontal flip, -1 for both | Yes | 
| Gamma correction | Applies gamma correction to the image.| *gamma*: Gamma value (should be between 0 and 2.5)| No |
| Gaussian blurring | Blurs an image using a Gaussian filter.| *kernel*: Kernel size for Gaussian blurring (either 3, 5, 7, 9 or 11).| No |
| Gaussian noise | Adds Gaussian noise to the image.  | *mean*: Mean value for Gaussian noise (an integer). *sigma*: Sigma value for Gaussian noise (an integer). | No |
| Invert | Inverts all values in images, i.e. sets a pixel from value v to 255-v | No parameters. | No |
| Median blurring | Blurs an image using the median filter. | *kernel*: Kernel size for median blurring (either 3, 5, 7, 9 or 11). | No |
| None | This augmentation technique does not change the image. | No parameters. | No |
| Raise blue channel | Raises the values in the blue channel. | *power*: Power for raising blue channel (value between 0.25 and 4) | No |
| Raise green channel | Raises the values in the green channel. | *power*: Power for raising green channel (value between 0.25 and 4) | No |
| Raise hue | Raises the hue value. | *power*: Power for raising hue channel (value between 0.25 and 4) | No |
| Raise red channel | Raises the value in the red channel. | *power*: Power for raising red channel (value between 0.25 and 4)| No |
| Raise saturation | Raises the saturation. | *power*: Power for raising saturation channel (value between 0.25 and 4) | No |
| Raise value | Raise the value of pixels. | *power*: Power for raising value channel (value between 0.25 and 4) | No |
| Resize | Resizes the image. | *percentage*: Percentage for resizing (double value). *method*: Method for resizing ("INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4","INTER_AREA") |  Yes |
| Rotate | Rotates the image. | *angle*: Angle for rotation (value between 0 and 360) | Yes |
| Salt and Pepper | Adds salt and pepper noise to the image. | *low*: Low value for salt and pepper (positive integer). *up*: Up value for salt and pepper (positive integer). | No |
| Sharpen | Sharpens the image. | No parameters. | No |
| Shift channels | Shifts the channels of the image. | "shift": Shifts input image channels in the range given (value between 0 and 1). | No |
| Shearing | Shears the image.| *a* : value for shearing (positive double). | Yes |
| Translation | Translates the image. | *x*: x transltation (integer). *y*: y translation (integer). | Yes |

