import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
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
from keras.utils import get_file
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from random import shuffle
from keras.layers import UpSampling2D

# model_resnet50_conv = ResNet50(include_top=False, pooling=None, weights='imagenet',input_shape=(228,304,3))
# model_resnet50_conv.layers.pop()
# print(model_resnet50_conv.summary())
from image_processing import save_depth_images_to_disk


class CNN:
    '''
    CNN classifier
    '''

    def __init__(self, train_x,train_y, test_x,test_y, epochs=50, batch_size=8):
        '''
        Initialize CNN classifier data
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x=train_x
        self.train_y = train_y
        self.test_x=test_x
        self.test_y = test_y
        # self.save_bottlebeck_features()
        self.train_data = np.load('bottleneck_features_train_resnet.npy')
        self.test_data = np.load('bottleneck_features_test_resnet.npy')
        # print(self.train_data.shape)
        # print(self.test_data.shape)

    def input_preprocessing(self, x):
        input = Input(shape=(480, 640, 3))
        input1 = MaxPool2D(pool_size=(2, 2), input_shape=(480, 640, 3))(input)
        print(input1.shape)
        input2 = Cropping2D(cropping=((6, 6), (8, 8)))(input1)
        print(input2.shape)

        model = Model(input, input2)
        x_dash = model.predict(x)
        return x_dash


    # model_resnet50_conv.summary()

    def save_bottlebeck_features(self):
        model = ResNet50(include_top=False,weights='imagenet')  # print(model.summary())
        model.layers.pop()
        model.outputs=[model.layers[-1].output]
        model.layers[-1].outbpound_nodes=[]
        print(model.summary())
        train_top = model.predict(self.input_preprocessing(self.train_x))
        print(train_top.shape)
        bottleneck_features_train = train_top
        np.save('bottleneck_features_train_resnet', bottleneck_features_train)

        test_top = model.predict(self.input_preprocessing(self.test_x))
        print(test_top.shape)
        bottleneck_features_test = test_top
        np.save('bottleneck_features_test_resnet', bottleneck_features_test)



    def train_top_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(1024, kernel_size=(1, 1),padding='same',
                              input_shape=self.train_data.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(512, kernel_size=(5, 5), padding='same'))
        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(256, kernel_size=(5, 5), padding='same'))
        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
        # self.model.add(UpSampling2D(size=(2, 2)))
        # self.model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))

        self.model.add(Conv2D(1, kernel_size=(2,7), padding='valid',activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.model.summary())

        return self.model


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
        return x_dash
        # n = x_dash.shape[0]
        # return x_dash.reshape((n, 4070))

    def delta(self, predicted_y):
        sum = 0
        for i in range(len(predicted_y)):
            rel_error = max((self.test_y[i] / predicted_y[i]), (self.test_y[i] / predicted_y[i]))
            if rel_error < 1.25:
                sum += 1
        return sum / len(predicted_y)


    def evaluate(self):
        '''
        test CNN classifier and get MSE
        :return: MSE, test_y, predicted_y
        '''
        self.estimator = KerasRegressor(build_fn=self.train_top_model, nb_epoch=self.epochs, batch_size=self.batch_size)
        self.train_y = self.label_preprocessing(self.train_y)
        self.test_y = self.label_preprocessing(self.test_y)

        self.estimator.fit(self.train_data, self.train_y, epochs=self.epochs)
        predicted_y = self.estimator.predict(self.test_data)
        MSE = mean_squared_error(self.test_y.ravel(), predicted_y.ravel())
        delta=self.delta(predicted_y)
        return MSE,delta self.test_y, predicted_y




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

    MSE,delta, transformed_test_y, predicted_y = cnn.evaluate()
    n_test = transformed_test_y.shape[0]
    print("Mean Squared Error  = {}".format(MSE))
    print("Delta  = {}".format(delta))


    try:
        # save_RGB_images_to_disk(test_x, "Test_X")
        save_depth_images_to_disk(transformed_test_y.reshape((n_test, 55, 74)), "Test_Y")
        save_depth_images_to_disk(predicted_y.reshape((n_test, 55, 74)), "Predicted_Y")
    except:
        np.save("transformed_test_y", transformed_test_y)
        np.save("predicted_y", predicted_y)
        print("Error in saving results to disk!!")


