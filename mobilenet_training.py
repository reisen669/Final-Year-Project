# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:41:57 2019
@author: Hor Sui Lyn 1161300122

This script is for MobileNet model training of Methods 1 and 2
   
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.optimizers import RMSprop 
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import tensorflow as tf
import math
import time

# constants
NO_OF_EPOCH = 20     
CLASS_NUM = 2
IMG_SIZE = 224      
BATCH_SIZE = 32   
BATCH_VAL = 16  
LEARNING_RATE = 0.0001
PRETRAINED = True # Set to False for Method 1, True for Method 2
FT = 10 # Only affects Method 2 (10, 20, 30)
NAME = "MobileNet: "

# file directories
TRAIN_DATA = 'C:/Users/AMD WINDOWS/Documents/MMU/YEAR 4/FYP/test/data/modified/train'
TEST_DATA = 'C:/Users/AMD WINDOWS/Documents/MMU/YEAR 4/FYP/test/data/modified/validation'
# related files in directories
categories = ["vNonPorn","vPorn"]

# name for .h5 and .json files
# Method 1: mobilenet_randomx_20_1, mobilenet_randomx1_20_1, mobilenet_randomx2_20_1
# Method 2: mobilenet_randomft10_20_1, mobilenet_randomft20_20_1, mobilenet_randomft30_20_1
FILE = "mobilenet_randomft10_20_1"

# record start time
start_time = time.time()

# In[2]: 
# Importing and building the required model

# load model without classifier layers
if PRETRAINED: # for Method 2 
    base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)) 
else: # for Method 1 
    base_model=MobileNet(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)) 

# unfreeze all layers 
base_model.trainable = True

# print number of layers in base model
print("Number of layers in the base model: ", len(base_model.layers)) # 87 for mobilenet

# freeze last few layers if required (for Method 2)
if PRETRAINED:
    fine_tune_at = len(base_model.layers) - FT
    print("fine_tune_at: ", fine_tune_at)
    n2 = "last " + str(FT) + " layers trainable"
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False
else:
    n2 = "all layers trainable"

# build model     
# Number of neurons: (64,8), (96,24), (128,36)
n1 = NAME + n2
model = Sequential(name=n1)
model.add(base_model)
model.add(GlobalAveragePooling2D()) # average over to convert the features to a single vector per image
model.add(Dense(96, 
                kernel_regularizer=regularizers.l1(0.005),
                activation='relu', input_dim=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Dropout(0.5))
model.add(Dense(24, 
                kernel_regularizer=regularizers.l1(0.005), 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(CLASS_NUM, activation='softmax'))

# check the architecture of model
model.summary()    

# In[3]:
# Loading the training (involves data augmentation) and validation data into the ImageDataGenerator

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

# passing class_mode='categorical' converts the labels to one hot encoded vectors
train_generator=train_datagen.flow_from_directory(TRAIN_DATA,
                                                 target_size=(IMG_SIZE,IMG_SIZE),     
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,             
                                                 class_mode='categorical',        
                                                 shuffle=True)      
validation_generator=validation_datagen.flow_from_directory(TEST_DATA,
                                                     target_size=(IMG_SIZE,IMG_SIZE),
                                                     color_mode='rgb',
                                                     batch_size=BATCH_VAL,
                                                     class_mode='categorical',
                                                     shuffle=True)      

# In[4]:
# Compile and train the model on the dataset

with tf.device('/gpu:0'):
    print("Training...")
    model.compile(optimizer=RMSprop(lr=LEARNING_RATE),loss='categorical_crossentropy',metrics=['accuracy']) 
    
    step_size_train=math.ceil(train_generator.samples / train_generator.batch_size)
    step_size_validation=math.ceil(validation_generator.samples / validation_generator.batch_size) 
    
    history = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=step_size_train,
                                 epochs=NO_OF_EPOCH, 
                                 validation_data=validation_generator,
                                 validation_steps=step_size_validation)   

# In[5]:
# Output history
    
print('\n# Evaluation ')
avg_loss, avg_acc, avg_val_loss, avg_val_acc = 0.0, 0.0, 0.0, 0.0
max_acc, min_acc = 0.0, 1.0

# compute average values    
for loss in history.history['loss']:
    avg_loss += loss
avg_loss /= NO_OF_EPOCH 
    
for acc in history.history['accuracy']:
    avg_acc += acc
    if acc > max_acc:
        max_acc = acc
    if acc < min_acc:
        min_acc = acc
avg_acc /= NO_OF_EPOCH 
    
for val_loss in history.history['val_loss']:
    avg_val_loss += val_loss
avg_val_loss /= NO_OF_EPOCH     
    
for val_acc in history.history['val_accuracy']:
    avg_val_acc += val_acc
avg_val_acc /= NO_OF_EPOCH    

print("Average training loss: ", avg_loss,
      "\nAverage training accuracy: ", avg_acc,
      "\nAverage validation loss: ", avg_val_loss, 
      " \nAverage validation accuracy: ", avg_val_acc, "\n")  
print("Max training accuracy: ", max_acc,
      "\nMin training accuracy: ", min_acc, "\n")     

# plot model accuracy graph
plt.plot(history.history['accuracy'], 'g.:')
plt.plot(history.history['val_accuracy'], 'r+-')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
# plot model loss graph
plt.plot(history.history['loss'], 'g.:')
plt.plot(history.history['val_loss'], 'r+-')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
   
# In[6]:
# Save model architecture and weights

# serialize model to JSON
model_json = model.to_json()  # architecture
with open(FILE+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(FILE+".h5")  # weights

# output model training time
t = time.time() - start_time
hr = math.floor(t/3600)
t = t - hr*60*60
minute = math.floor(t/60)
sec = t - minute*60
print("--- ", hr, " hours ", minute, " minutes ", int(sec), " seconds---")

# In[7]:
# Making predictions

pred_generator=validation_datagen.flow_from_directory(TEST_DATA,
                                                     target_size=(IMG_SIZE,IMG_SIZE),
                                                     color_mode='rgb',
                                                     batch_size=BATCH_VAL,
                                                     class_mode='categorical',
                                                     shuffle=False)      

predictions = model.predict_generator(pred_generator, steps=step_size_validation)

# get predicted class
predicted_classes = np.argmax(predictions, axis=1)
# get true class
true_classes = pred_generator.classes
# get class labels
class_labels = list(pred_generator.class_indices.keys())   

# generate report and output accuracy score
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
print("Accuracy score: ", accuracy_score(true_classes, predicted_classes))

# print confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
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
