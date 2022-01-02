import tensorflow as tf
import tensorflow.keras as keras


class MnistEncoder(tf.keras.Model):

    def __init__(self):
        super(MnistEncoder, self).__init__()
        self.conv_1 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="sigmoid")
        self.conv_2 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="sigmoid")
        self.conv_3 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="sigmoid")
        self.conv_4 = keras.layers.Conv2D(10, 3, 2, padding='same', activation="sigmoid")
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x, self.flatten(x)
