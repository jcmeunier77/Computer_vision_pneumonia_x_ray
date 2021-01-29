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


<p align="center">
    <img src="https://github.com/jcmeunier77/prediction_API/blob/master/img_out/pc%20ranked%20by%20prices.png">
</p>

#### 4. Modelization
- [x] Features : 
  - type of building: house/apartment
  - living area: square meters
  - field's surface: square meters
  - number of facades
  - number of bedrooms
  - garden: yes/no
  - terrace: yes/no
  - terrace area: square meters
  - equipped kitchen: yes/no
  - fireplace: yes/no
  - swimming pool: yes/no
  - state of the building: as new, just renovated, good, to refresh, to renovate, to restore (one hot encoding)
- [x] Target: 
  - House price: euros 
- [x] Machine learning model: 
  - Multiple models using increasing number of features and based on various algorithm (i.a. linear, SVM, decision tree, XGBoost) were trained and evaluated.
  - The best model was based on the XGBoost algorithm (n_estimators=700, max_depth= 4, learning_rate= 0.3) and provided an r_square coefficient of .82 on the train set and of .76 on the test set
  - The best fitted model was save as a pickel file which was integrated in the API for price estimation 
  - Examples of python code for data manipulation and algorithms development are stored in the [notebook folder](https://github.com/jcmeunier77/prediction_API/tree/master/notebooks%20data%20preparation%20and%20ML%20algorithms) of the current repository

### Project output
#### 1. API Structure 

<p align="center">
    <img src="https://github.com/jcmeunier77/prediction_API/blob/master/img_out/API%20structure.png">
</p>

### 2. API Routes
- [x] Estimate: in

<p align="center">
    <img src="https://github.com/jcmeunier77/prediction_API/blob/master/img_out/API%20estimate%20in.png">
</p>

- [x] Estimate: out
