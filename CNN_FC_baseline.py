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
from sklearn.metrics import mean_squared_error
from random import shuffle

try:
    from image_processing import save_depth_images_to_disk, \
    save_RGB_images_to_disk
except:
    pass

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epochs = 50, batch_size = 8):

        '''
        Initialize CNN classifier data
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x=train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


    def make_model(self):
        '''
        Build CNN classifier model architecture
        '''
        input_shape = (480, 640, 3)

        self.model = Sequential()
        self.model.add(MaxPool2D(pool_size=(2, 2), input_shape=input_shape))
        self.model.add(Cropping2D(cropping=((6, 6), (8, 8))))
        self.model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4),
                              activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, kernel_size=(5, 5),padding='same',
                              activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(384, kernel_size=(3, 3),padding='same',
                              activation='relu'))
        self.model.add(Conv2D(384, kernel_size=(3, 3),padding='same',
                              activation='relu'))
        self.model.add(Conv2D(256, kernel_size=(3, 3), strides=(2,2),
                              activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(74 * 55, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.model.summary())

        return self.model


    def preprocessing(self,x):

        input=Input(shape=(480,640,1))
        input1=MaxPool2D(pool_size=(2, 2),input_shape=(480,640,1))(input)
        # print(input1.shape)
        input2=Cropping2D(cropping=((6, 6), (8, 8)))(input1)
        print(input2.shape)
        input3=MaxPool2D(pool_size=(4, 4))(input2)
        # print(input3.shape)
        input4=Cropping2D(cropping=((1, 1), (1, 1)))(input3)
        print(input4.shape)
        model=Model(input,input4)
        x_dash=model.predict(x)
        n = x_dash.shape[0]
        return x_dash.reshape((n, 4070))


    def evaluate(self):
        '''
        test CNN classifier and get MSE
        :return: MSE, test_y, predicted_y
        '''
        self.estimator = KerasRegressor(build_fn=self.make_model, nb_epoch=self.epochs, batch_size=self.batch_size)
        self.train_y = self.preprocessing(self.train_y)
        self.test_y = self.preprocessing(self.test_y)
        self.estimator.fit(self.train_x, self.train_y, epochs= self.epochs)
        predicted_y = self.estimator.predict(self.test_x)
        MSE = mean_squared_error(self.test_y, predicted_y)
        return MSE, self.test_y, predicted_y

if __name__ == '__main__':

    depth = np.load('depth_n_h_w_1.npy')
    # print(data.train_x[:10])
    r_g_b = np.load('images_n_h_w_c.npy')

    image_indices=[i for i in range(len(r_g_b))]
    # shuffle(image_indices)

    split_index = int(0.8 * len(image_indices))
    train_x = r_g_b[:split_index]
    train_y=depth[:split_index]

    test_x=r_g_b[split_index:]
    test_y=depth[split_index:]

    # baseline=CNN(train_x,train_y,test_x,test_y)

    cnn = CNN(train_x,train_y,test_x,test_y)
    # cnn.preprocessing(train_y)
    # cnn.make_model()
    MSE, transformed_test_y, predicted_y = cnn.evaluate()
    n_test = transformed_test_y.shape[0]
    print("Mean Squared Error  = {}".format(MSE))

    try:
        # save_RGB_images_to_disk(test_x, "Test_X")
        save_depth_images_to_disk(transformed_test_y.reshape((n_test, 55, 74)), "Test_Y")
        save_depth_images_to_disk(predicted_y.reshape((n_test, 55, 74)), "Predicted_Y")
    except:
        np.save("transformed_test_y", transformed_test_y)
        np.save("predicted_y", predicted_y)
        print("Error in saving results to disk!!")

