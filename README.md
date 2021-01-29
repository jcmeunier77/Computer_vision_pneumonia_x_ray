# Computer_vision_pneumonia_x_ray

Authors of the project : [Kai Yung TAN (Adam)](https://github.com/kaiyungtan) & [Jean Christophe Meunier](https://github.com/jcmeunier77) 

## 1. Purpose and project objective 

### Purpose 
- [x] Learning how to design and evaluate a custom made convolutional neural network for practical purposes
- [x] Using CNN models to analyse x ray images
- [x] Designing a CNN capable of recognising pneumonia in x-rays of patients


### Objectives 

- [x] Consolidate the knowledge in Python, specifically in : Tensorflow/kerras, NumPy, Pandas, Matplotlib,...
- [x] To be able to search and implement new librairies
- [x] Consolidate knowledge of data science and machine/deep learning algorithm for developping an accurate regression prediction model
- [x] To be able perform appropriate model hyperparametrisation

### Features 
#### Must-have 
- [x] A CNN trained on a large x ray dataset (>5k) that can recognise new images outside of the training set
- [x] Proper model evaluation (split dataset, confusion matrix, etc)
- [x] Visualisations of model results (properly labeled, titled...)

#### Nice-to-Have
- [x] A visualisation of the feature maps of the model
- [x] Comparison with other CNN model structures
- [x] Assessing and comparing

### Context of the project 
- [x] All the work achieved was done during the BeCode's AI/data science bootcamp 2020-2021

## 2. The project 
### Working plan and steps 
#### 1. Research 
- [x] Research and understand the term, concept and requirement of the project.
- [x] Discover new libraries that can serve the project purposes 
- [x] Developing, using and testing machine learning algorithm (i.a. tensorflow/kerras,...)
- [x] Consolidating knowledge on model building and model hyperparametrisation (e.g. type of layers, pooling, dropout, batch normalization, type of activation functions,...)
- [x] Data augmentation
- [x] Aside from that, we also searched documentation on the internet on existing published work and/or studies on x ray data manipulation and modelization, as for example : 
  - [sibeltan/pneumonia_detection_CNN](https://github.com/sibeltan/pneumonia_detection_CNN)
  - [Jain et al., 2020. Pneumonia detection in chest X-ray images using convolutional neural networks and transfer learning. Measurement, 165, 1.](https://www.sciencedirect.com/science/article/abs/pii/S0263224120305844)

#### 2. Data collection 
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

- [x] Examples of data input 

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/0.%20sample%20xrays.png">
</p>
      
#### 3. Data manipulation 
- [x] Image size reduction: original jpg were reduced to size 128 x 128 in order to accelerate data processing during models training

- [x] Standardisation of the images 

- [x] Data augmentation using CV2 library and the 'ImageDataGenerator' function in order to increase training quality

#### 4. Modelization
In total, a number of 17 models were build, trained and compared using various hyperparametrisation ([see notebook section](https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/tree/master/notebook%20with%20computed%20CNN%20models):
- [x] depth of the neural network
- [x] type of layers (dense, convolutional,...)
- [x] filters 
- [x] type of activation (i.a. relu, leakyrelu, sigmoid, softmax,...)
- [x] dropout 
- [x] pooling 
- [x] batch normalization

For each model, hyperparametrisation was fine-tuned based on the performance indices on the test data set (624 pictures). When a model reached a satifying accuracy, he was finally rerun on the validation set (16 pictures)

The best fitted model was choosen partly based on previous good performance on train and test data set but mostly on performance on validation data set.  

### Final best fitting model
#### 1. Model architecture  
- 8 convolution layers (filters=32/32/32/64/64/64/128/128, kernel_size=(3, 3) activation='Leaky-relu')
- MaxPool2D((2, 2)
- Dropout(0.25) on all layers excepting the last one
- Flatten
- 1 dense layer (1024, activation='relu')
- model.add(Dense(2, activation='sigmoid'))
- Dropout(0.5)
- loss='binary_crossentropy', optimizer='adam'
- shuffle = True
- data augmentation: rotation_range = 20, zoom_range = 0.2, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = True, vertical_flip = True
- Batch size : 16
- Epochs : 100

### 2. Performance evaluation
- [x] Loss and accuracy

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/1.%20final%20loss%20accuracy.png">
</p>

- [x] Confusion matrix on test set

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/3.%20final%20test%20confusion%20matrix.png">
</p>

- [x] Performance indices on test set

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/4.%20final%20test%20perfomance%20indicators.png">
</p>

- [x] Confusion matrix on validation set

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/5.%20final%20val%20confusion%20matrix.png">
</p>

- [x] Performance indices on validation set

<p align="center">
    <img src="https://github.com/jcmeunier77/Computer_vision_pneumonia_x_ray/blob/master/img/6.%20final%20val%20perfomance%20indicators.png">
</p>
