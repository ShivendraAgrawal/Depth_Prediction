import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import applications
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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from random import shuffle


# model = applications.VGG16(include_top=False, weights='imagenet')
#
# print(model.summary())

class CNN:
    '''
    CNN classifier
    '''

    def __init__(self, train_x,train_y, test_x,test_y, epochs=1, batch_size=3):
        '''
        Initialize CNN classifier data
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x=train_x
        self.train_y = train_y
        self.test_x=test_x
        self.test_y = test_y

        # Running some functions in init
        # self.save_bottlebeck_features()

    def train_top_model(self):
        top_model_weights_path = None

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(4096, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(74 * 55, activation='relu'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(top_model_weights_path)

        # add the model on top of the convolutional base
        model.add(top_model)

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:25]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))

        return model

    def input_preprocessing(self, x):
        input = Input(shape=(480, 640, 3))
        input1 = MaxPool2D(pool_size=(2, 2), input_shape=(480, 640, 3))(input)
        print(input1.shape)
        input2 = Cropping2D(cropping=((8, 8), (48, 48)))(input1)
        print(input2.shape)

        model = Model(input, input2)
        x_dash = model.predict(x)
        print(x_dash.shape)
        # n = x_dash.shape[0]
        return x_dash


    def label_preprocessing(self, x):
        input = Input(shape=(480, 640, 1))
        input1 = MaxPool2D(pool_size=(2, 2), input_shape=(480, 640, 1))(input)
        # print(input1.shape)
        input2 = Cropping2D(cropping=((6, 6), (8, 8)))(input1)
        print(input2.shape)
        input3 = MaxPool2D(pool_size=(4, 4))(input2)
        # print(input3.shape)
        input4 = Cropping2D(cropping=((1, 1), (1, 1)))(input3)
        print(input4.shape)
        model = Model(input, input4)
        x_dash = model.predict(x)
        n = x_dash.shape[0]
        return x_dash.reshape((n, 4070))

    def evaluate(self):
        '''
        test CNN classifier and get MSE
        :return: MSE, test_y, predicted_y
        '''
        self.estimator = KerasRegressor(build_fn=self.train_top_model, nb_epoch=self.epochs, batch_size=self.batch_size)
        self.train_y = self.label_preprocessing(self.train_y)
        self.test_y = self.label_preprocessing(self.test_y)
        self.train_data=self.input_preprocessing(self.train_x)
        self.test_data=self.input_preprocessing(self.test_x)

        self.estimator.fit(self.train_data, self.train_y,epochs=self.epochs)
        predicted_y = self.estimator.predict(self.test_data)
        MSE = mean_squared_error(self.test_y, predicted_y)
        return MSE, self.test_y, predicted_y




if __name__ == '__main__':

    depth = np.load('depth_n_h_w_1.npy')
    # print(data.train_x[:10])
    r_g_b = np.load('images_n_h_w_c.npy')

    image_indices = [i for i in range(len(r_g_b))]
    # shuffle(image_indices)

    split_index = int(0.8 * len(image_indices))
    train_x = r_g_b[:split_index]
    train_y = depth[:split_index]

    test_x = r_g_b[split_index:]
    test_y = depth[split_index:]

    # baseline=CNN(train_x,train_y,test_x,test_y)

    cnn = CNN(train_x, train_y, test_x, test_y)
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

