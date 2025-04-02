# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 19:32:56 2020  
@author: Hor Sui Lyn 1161300122

This script is for Methods 3 and 4
Models trained using Methods 1 and 2 will be loaded to extract features
SVM classifier will be used to classify the data

"""

# import libraries
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import math
from keras.models import model_from_json
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
from keras.models import Model
from sklearn import svm
import itertools

# file directories
TRAIN_DATA = 'C:/Users/AMD WINDOWS/Documents/MMU/YEAR 4/FYP/test/data/modified/train'
TEST_DATA = 'C:/Users/AMD WINDOWS/Documents/MMU/YEAR 4/FYP/test/data/modified/validation'

# name of .h5 and .json files to be loaded for the CNN model
# Method 1: mobilenet_randomx1_20_1, vgg19_randomx_20_1, resnet50_randomx2_20_1
# Method 2: mobilenet_randomft30_20_1, vgg19_randomft5_20_1, resnet50_randomft20_20_1
FILE = "resnet50_randomft20_20_1"

# constants
LEARNING_RATE = 0.0001
BATCH_VAL = 16
BATCH_SIZE = 32
IMG_SIZE = 224

if (str(FILE)).find("mobilenet") != -1:
    MODEL = 'mobilenet'
    from keras.applications.mobilenet import preprocess_input 
elif (str(FILE)).find("resnet50") != -1:
    MODEL = 'resnet50'
    from keras.applications.resnet_v2 import preprocess_input 
elif (str(FILE)).find("vgg19") != -1:
    MODEL = 'vgg19'
    from keras.applications.vgg19 import preprocess_input 
print("model: ", MODEL)

# limit gpu memory usage
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# record start time
start_time = time.time()

# In[2]:

# load json and create model
json_file = open(FILE + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(FILE + ".h5")
print("Loaded model from disk")

## check the architecture of model
model.summary()

# set output layers of CNN models where features extracted will be fed into SVM
if MODEL=='resnet50':
    if (str(FILE)).find("ft") != -1:
        model = Model(inputs=model.input, outputs=model.get_layer("dropout_3").output)
    else:
        model = Model(inputs=model.input, outputs=model.get_layer("dropout_1").output)
elif MODEL == 'vgg19':
    if (str(FILE)).find("ft") != -1:
        model = Model(inputs=model.input, outputs=model.get_layer('dropout_7').output)
    else:
        model = Model(inputs=model.input, outputs=model.get_layer('dropout_9').output)
elif MODEL == 'mobilenet':
    if (str(FILE)).find("ft") != -1:
        model = Model(inputs=model.input, outputs=model.get_layer('dropout_3').output)
    else:
        model = Model(inputs=model.input, outputs=model.get_layer('dropout_1').output) 

# compile model
model.compile(optimizer=RMSprop(lr=LEARNING_RATE),loss='categorical_crossentropy',metrics=['accuracy'])

# In[3]:

# loading the training (involves data augmentation) and validation data into the ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=90, 
                                   shear_range=0.2,
                                   zoom_range=0.3, 
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   brightness_range=[0.9,1.5], 
                                   width_shift_range=0.5,  
                                   height_shift_range=0.5,  
                                   preprocessing_function=preprocess_input)    
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator=validation_datagen.flow_from_directory(TRAIN_DATA,
                                                     target_size=(IMG_SIZE,IMG_SIZE),
                                                     color_mode='rgb',
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical',
                                                     shuffle=False)      
pred_generator=validation_datagen.flow_from_directory(TEST_DATA,
                                                     target_size=(IMG_SIZE,IMG_SIZE),
                                                     color_mode='rgb',
                                                     batch_size=BATCH_VAL,
                                                     class_mode='categorical',
                                                     shuffle=False)

step_size_train = math.ceil(train_generator.samples / train_generator.batch_size)                                                     
step_size_validation = math.ceil(pred_generator.samples / pred_generator.batch_size)

features=[]
y = []
features_test = []    
y_test = []
class_labels = []

# extract features from training data       
features = model.predict_generator(train_generator, steps=step_size_train)
print("Feature: ", np.size(features))

# get labels of training data
y = train_generator.classes

# In[4]:
# Reshaping

f = np.array(features)
dimension = int(np.size(features) / (train_generator.samples)) 
print(dimension)
features = f.reshape((train_generator.samples), dimension)

# training support vector machine as classifier
with tf.device('/gpu:0'):
    print("Matching...")
    svclassifier = svm.SVC(gamma='scale') #Support Vector Classification
    svclassifier.fit(features, y)
    del features

# In[5]:
# Predicting
    
with tf.device('/gpu:0'):
    features_test = model.predict_generator(pred_generator, steps=step_size_validation)
    print("Features_test: ", np.size(features_test))
    y_test = pred_generator.classes 
    class_labels = list(pred_generator.class_indices.keys())   

    f_test = np.array(features_test)
    dimension = int(np.size(features_test) / (pred_generator.samples)) 
    features_test = f_test.reshape((pred_generator.samples),dimension)
    
    print("Predicting...")
    y_pred = svclassifier.predict(features_test)
    
# In[6]:
# Output time required
    
t = time.time() - start_time
hr = math.floor(t/3600)
t = t - hr*60*60
minute = math.floor(t/60)
sec = t - minute*60
print("--- ", hr, " hours ", minute, " minutes ", int(sec), " seconds---")

# In[7]:

# generate report and output accuracy score
print(classification_report(y_test,y_pred,target_names=class_labels))
print("Accuracy score: ", accuracy_score(y_test, y_pred))

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
def plot_confusion_matrix(cm, class_labels, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation = 45)
    plt.yticks(tick_marks, class_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')    
    print(cm)
    
    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

cm_plot_labels = ['NonPorn', 'Porn']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
