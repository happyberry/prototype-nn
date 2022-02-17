import tensorflow as tf
import tensorflow.keras as keras


class RpsEncoder(tf.keras.Model):

    def __init__(self):
        super(RpsEncoder, self).__init__()
        self.conv_1 = keras.layers.Conv2D(32, 5, 2)
        self.leaky_relu = keras.layers.LeakyReLU()
        self.conv_2 = keras.layers.Conv2D(10, 5, 2, activation="sigmoid")
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        return x, self.flatten(x)
