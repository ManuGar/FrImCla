{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"Readme.md","version":"0.3.2","views":{},"default_view":{},"provenance":[]}},"cells":[{"metadata":{"id":"J8Imb1ms5MNu","colab_type":"text"},"cell_type":"markdown","source":["# FrImCla"]},{"metadata":{"id":"P01XMT1x5OyL","colab_type":"text"},"cell_type":"markdown","source":["FrImCla is an open-source framework for Image Classification using traditional and deep learning tchniques. It supports a wide variety of deep learning and classical computer vision models. "]},{"metadata":{"id":"FwCcrG765cgj","colab_type":"text"},"cell_type":"markdown","source":["## Features"]},{"metadata":{"id":"pTSj-hXw5cjj","colab_type":"text"},"cell_type":"markdown","source":["*   Several deep learning models\n","*   Classical computer vision techniques\n","*   Different performance measures\n","*   Jupyter notebook to use the framework\n","*   Ability to test several models at the same time\n","*   It is possible to easily add models and apply them into the framework to comprobe the performance over the dataset"]},{"metadata":{"id":"xT1Q-uT85cmL","colab_type":"text"},"cell_type":"markdown","source":["## Requeriments and installation of the library"]},{"metadata":{"id":"ny8fs5A75coz","colab_type":"text"},"cell_type":"markdown","source":["The library uses Python 2.7, which must be installed. \n","The following packages must be installed:\n","\n","\n","*   numpy\n","*   scipy\n","*   scikit_learn\n","*   scikit-image\n","*   keras\n","*   h5py\n","*   OpenCV (i.e. cv2 must be available in python)\n","*   commentjson"]},{"metadata":{"id":"Cly2uA4N5crc","colab_type":"text"},"cell_type":"markdown","source":["# Documentation"]},{"metadata":{"id":"4Xz3GNXM5cuE","colab_type":"text"},"cell_type":"markdown","source":["Aquí poner el enlace al readmen de como usarlo y demás"]},{"metadata":{"id":"PEyb5NAv5cws","colab_type":"text"},"cell_type":"markdown","source":["## Examples"]},{"metadata":{"id":"3gYj-O8V5czb","colab_type":"text"},"cell_type":"markdown","source":["Aquí poner los diferentes enlaces a los cuadernos"]},{"metadata":{"id":"y370YEaZ-b0c","colab_type":"text"},"cell_type":"markdown","source":["## List of feature extractor models"]},{"metadata":{"id":"DAJHuG_ny2hs","colab_type":"text"},"cell_type":"markdown","source":["| Model | Description | Parameters | \n","| --- | ------- | --- | --- |\n","| VGG16| Keras model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition. It has been obtained by directly converting the Caffe model provived by the authors. | *include_top*(either True or False). |\n","| VGG19 | Keras model of the 19-layer network used by the VGG team in the ILSVRC-2014 competition. It has been obtained by directly converting the Caffe model provived by the authors.| *include_top*(either True or False).  |\n","| Resnet |Residual neural networks utilizes skip connections or short-cuts to jump over some layers. In its limit as ResNets it will only skip over a single layer. With an additional weight matrix to learn the skip weights it is referred to as HighwayNets. With several parallel skips it is referred to as DenseNets. | *include_top*(either True or False).  |\n","| Inception | The network used a CNN inspired by LeNet but implemented a novel element which is dubbed an inception module. It used batch normalization and image distortions. This module is based on several very small convolutions in order to drastically reduce the number of parameters. | *include_top*(either True or False).\n","| Googlenet |  This network used a new variant of convolutional neural network called “Inception” for classification, and for detection the R-CNN was used. Google’s team was able to train a much smaller neural network and obtained much better results  compared to results obtained with convolutional neural networks in the previous year’s challenges. | No parameters. |\n","| Overfeat | It is a combination of CNN and another machine learning classifier ( LR, SVM, etc.). | *output_layers*(by default [-3]).  |\n","| Xception | This architecture has 36 convolutional layers forming the feature extraction base of the network. The Xception architecture is a linear stack of depthwise separable convolution layers with residual connections.| *include_top*(either True or False). \n","| Densenet | This is a stack of dense blocks followed by transition layers. Each block consists of a series of units. Each unit packs two convolutions, each preceded by Batch Normalization and ReLU activations. Besides, each unit outputs a fixed number of feature vectors. This parameter, described as the growth rate, controls how much new information the layers allow to pass through. | No parameters |\n","| LAB888 | Histogram of the images with the lab colour format with 8 bins| No parameters. |\n","| LAB444 | Histogram of the images with the lab colour format with 4 bins | *bins*(by default 4,4,4). |\n","| HSV888 | Histogram of the images with the hsv colour format with 8 bins | No parameters. |\n","| HSV444 | Histogram of the images with the hsv colour format with 4 bins |*bins*(by default 4,4,4). | \n","| Haralick | This model suggested the use of gray level co-occurrence matrices (GLCM). This method is based on the joint probability distributions of pairs of pixels. GLCM show how often each gray level occurs at a pixel located at a fixed geometric position relative to each other pixel, as a function of the gray level. An essential component is the definition of eight nearest-neighbor resolution cells that define different matrices for different angles (0°,45°,90°,135°) and distances between the horizontal neighboring pixels. | No parameters. | \n","| Hog | The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy. | No parameters. |\n"]},{"metadata":{"id":"ML9Ax-MC-cLj","colab_type":"text"},"cell_type":"markdown","source":["## List of classifier models"]},{"metadata":{"id":"r9tvaxSH-cdb","colab_type":"text"},"cell_type":"markdown","source":["*   Multilayer perceptron (MLP)\n","*   Support vector machine (SVM)\n","*   KNN  https://www.techopedia.com/definition/32066/k-nearest-neighbor-k-nn \n","*   Logistic Regression\n","*   Random Forest"]},{"metadata":{"id":"DIMjPRqvyKjn","colab_type":"text"},"cell_type":"markdown","source":["| Classifier | Description | Parameters | \n","| --- | --- | --- | --- |\n","| Multilayer perceptron (MLP)| | |\n","| Support vector machine (SVM)| | |\n","| K-Nearest Neighbor(KNN) | | |\n","| Logistic Regression (LR)| | |\n","| Random Forest (RF)| | |\n"]},{"metadata":{"id":"49xI9xPcybHQ","colab_type":"code","colab":{"autoexec":{"startup":false,"wait_interval":0}}},"cell_type":"code","source":[""],"execution_count":0,"outputs":[]}]}