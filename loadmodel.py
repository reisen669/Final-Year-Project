# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:58:18 2020
@author: Hor Sui Lyn 1161300122

This script is for loading of CNN models and classification of video frames for F-Filter

"""

# import libraries
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import math
from keras.models import model_from_json
from keras.optimizers import RMSprop 
from keras.applications.mobilenet import preprocess_input
from sklearn import svm
import os
from pathlib import Path
from keras.models import Model

# constants
IMG_SIZE = 224      
BATCH_VAL = 16    
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# file directory
TRAIN_DATA = 'C:/Users/AMD WINDOWS/Documents/MMU/YEAR 4/FYP/test/data/modified/train'

# depends on method, and also model used
MODEL = ["mobilenet",
         "vgg19",
         "resnet50"
        ]

# load respective CNN model
def load(method, path):
    name = ""
    model = ""
    svmbool = False
    if method == 1:    
        name = "mobilenet_randomx1_20_1" # MobileNet with (96, 24) neurons
        model = MODEL[0]
    elif method == 2:
        name = "resnet50_randomft20_20_1" # ResNet50_V2 with (128, 36) neurons, 20 layers fine-tuned
        model = MODEL[2] 
    elif method == 3:
        name = "mobilenet_randomx1_20_1" # MobileNet with (96, 24) neurons
        svmbool = True
        model = MODEL[0] 
    elif method == 4:
        name = "resnet50_randomft20_20_1" # ResNet50_V2 with (128, 36) neurons, 20 layers fine-tuned
        svmbool = True
        model = MODEL[2] 
        
    json = name + ".json"
    h5 = name + ".h5"
    imgsize = 224    
    
    pre = os.getcwdb()
    pre = Path(pre.decode('utf-8'))
    result = loadrespectivemodel(json, h5, imgsize, pre, path, svmbool, model, name)
    return result

def loadrespectivemodel(json, h5, imgsize, pre, category, svmbool, mod, name):
    # load json and create model
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(h5)
    print("Loaded model from disk")
    print(model.summary())

    c = []
    c.append(category)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)     
    validation_generator=validation_datagen.flow_from_directory(pre,
                                                         target_size=(imgsize,imgsize),
                                                         color_mode='rgb',
                                                         batch_size=BATCH_VAL,
                                                         class_mode='categorical',
                                                         classes=c,
                                                         shuffle=False)       # keep data in same order as labels    
    step_size_validation=math.ceil(validation_generator.samples / validation_generator.batch_size) 

    # compile model and make predictions    
    with tf.device('/gpu:0'):
        # Extract features from model if SVM classifier is used
        if svmbool:
            if mod == MODEL[2]: #'resnet50' 
                if (str(name)).find("ft") != -1:
                    model = Model(inputs=model.input, outputs=model.get_layer("dropout_4").output)
                else:
                    model = Model(inputs=model.input, outputs=model.get_layer("dropout_1").output)
            elif mod == MODEL[1]: #'vgg19'
                if (str(name)).find("ft") != -1:
                    model = Model(inputs=model.input, outputs=model.get_layer('dropout_1').output)
                else:
                    model = Model(inputs=model.input, outputs=model.get_layer('dropout_11').output)
            elif mod == MODEL[0]: #'mobilenet'
                if (str(name)).find("ft") != -1:
                    model = Model(inputs=model.input, outputs=model.get_layer('dropout_11').output)
                else:
                    model = Model(inputs=model.input, outputs=model.get_layer('dropout_1').output) 
        
        model.compile(optimizer=RMSprop(lr=LEARNING_RATE),loss='categorical_crossentropy', metrics=['accuracy'])
        predictions = model.predict_generator(validation_generator, steps=step_size_validation)
        
        if svmbool:
            features=[]
            features_test = []
            train_generator=validation_datagen.flow_from_directory(TRAIN_DATA,
                                                     target_size=(IMG_SIZE,IMG_SIZE),
                                                     color_mode='rgb',
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical',
                                                     shuffle=False)      
            step_size_train = math.ceil(train_generator.samples / train_generator.batch_size)  
            features = model.predict_generator(train_generator, steps=step_size_train)
            print("Feature: ", np.size(features))
            y = train_generator.classes
            f = np.array(features)
            dimension = int(np.size(features) / train_generator.samples)
            print(dimension)
            features = f.reshape(train_generator.samples, dimension)
            svclassifier = svm.SVC(gamma='scale') #Support Vector Classification
            svclassifier.fit(features, y)
            del features
            
            print("Features_test: ", np.size(predictions))
            f_test = np.array(predictions)
            dimension = int(np.size(predictions) / validation_generator.samples)
            features_test = f_test.reshape(validation_generator.samples, dimension)
            
            print("Predicting...")
            predicted_classes = svclassifier.predict(features_test)
        else:
            predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes