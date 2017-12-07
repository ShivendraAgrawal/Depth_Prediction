import keras
from keras import Input, optimizers
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from random import shuffle
import os
from matplotlib import pyplot as plt
from PIL import Image
import os

os.mkdir("images")

if __name__ == '__main__':

    depth = np.load('depth_n_h_w_1.npy')
    # print(data.train_x[:10])
    r_g_b = np.load('images_n_h_w_c.npy')

    image_indices=[i for i in range(len(r_g_b))]
    # shuffle(image_indices)
    print(len(r_g_b))
    split_index = int(0.01*len(image_indices))
    train_x = r_g_b[:split_index]
    train_y=depth[:split_index]

    test_x=r_g_b[split_index:]
    test_y=depth[split_index:]
    img = []
    depth = []

    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 zca_whitening=True,
                                 zca_epsilon=1e-8,
                                 rotation_range=0,
                                 width_shift_range=0,
                                 height_shift_range=0,
                                 horizontal_flip=True)



    datagen.fit(train_x)
    datagen.fit(train_y)

    for epoch in range(1):
        batches = 0
        for batch in datagen.flow(train_x,seed=2011,batch_size=2,save_to_dir='images', save_prefix='aug', save_format='png'):
            img.append(batch)
            batches += 1
            if batches >= 2:
                break
        batches = 0
        for batch in datagen.flow(train_y,seed=2011, batch_size=2, save_to_dir='images',save_prefix='aug_depths', save_format='png'):
            depth.append(batch)
            batches += 1
            if batches >= 2:
                break
        print('Augmenting Depth now')