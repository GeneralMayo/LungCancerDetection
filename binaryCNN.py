from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import regularizers

from scipy import ndimage
from PIL import Image
import numpy as np

import math
from keras import backend as K
from keras.utils import plot_model
import os

def load_data():
    #get refference data
    dataFolder = 'Data/ActiveData/'


    trainB = []
    trainM = []
    testB = []
    testM = []
    for root, dirs, files in os.walk(dataFolder):
        if(len(files) > 1):
            print(root)
            #find what data is being loaded
            testTrain = root.split('/')[-2]
            classType = root.split('/')[-1]
            if(testTrain == 'Train' and classType == 'Benign'):
                listToLoad = trainB
            elif(testTrain == 'Train' and classType == 'Malignant'):
                listToLoad = trainM
            elif(testTrain == 'Test' and classType == 'Benign'):
                listToLoad = testB
            else:
                listToLoad = testM

            for fileName in files:
                listToLoad.append(np.load(os.path.join(root,fileName)))
                
    trainData = np.array(trainB+trainM)
    trainLabels = np.ones(len(trainB)+len(trainM))
    trainLabels[0:len(trainB)] = np.zeros(len(trainB))

    testData = np.array(testB+testM)
    testLabels = np.ones(len(testB)+len(testM))
    testLabels[0:len(testB)] = np.zeros(len(testB))

    return trainData, trainLabels, testData, testLabels

trainData, trainLabels, testData, testLabels = load_data()

batch_size = 32

model = Sequential()
model.add(Conv2D(24, (5, 5), input_shape=trainData[0].shape, data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3))), 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
		#featurewise_center=True,
		#featurewise_std_normalization=True,
        #preprocessing_function=filter_image,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=40,
        data_format="channels_first",
        rescale=1./255)

test_datagen = ImageDataGenerator(
	#featurewise_center=True,
	#featurewise_std_normalization=True,
    #preprocessing_function=filter_image,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    rotation_range=40,
	data_format="channels_first",
	rescale=1./255)


train_generator = train_datagen.flow(
        trainData,
        trainLabels,
        batch_size = batch_size)


validation_generator = test_datagen.flow(
        testData,
        testLabels,
        batch_size = batch_size)

model.fit_generator(
        train_generator,
        callbacks=[ModelCheckpoint('BinaryCNN.h5', monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='./logs', write_graph=True, write_images=False, histogram_freq=10)],
            #EarlyStopping(monitor='val_loss', patience=4)],
        steps_per_epoch=20000//batch_size,
        epochs=80,
        validation_data=validation_generator,
        validation_steps=1000//batch_size)


#model.save('Problem1_CNN.h5')
