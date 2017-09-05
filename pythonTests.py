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


#get x and y arrays
BenignFolder = 'Data/ActiveData/Train/Benign'
reffFileName = os.listdir(BenignFolder)[0]
reffData = np.load(os.path.join(BenignFolder,reffFileName))
allDataShape = tuple([60]+list(reffData.shape))
allData = np.zeros(allDataShape)

fileNames = os.listdir(BenignFolder)
for idx in range(60):
	allData[idx,:,:,:] = np.load(os.path.join(BenignFolder,fileNames[idx]))

labels = np.zeros(60)

train_datagen = ImageDataGenerator(
		#featurewise_center=True,
		#featurewise_std_normalization=True,
        #preprocessing_function=filter_image,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=90,
        data_format="channels_first",
        rescale=1./255)

train_generator = train_datagen.flow(
		allData,
		labels)

(x,y) = train_generator.next()

print(x.shape,y.shape)