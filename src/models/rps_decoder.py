import tensorflow as tf
import tensorflow.keras as keras


class RpsDecoder(tf.keras.Model):

    def __init__(self):
        super(RpsDecoder, self).__init__()
        self.deconv_1 = keras.layers.Conv2DTranspose(10, 5, 2, output_padding=1)
        self.leaky_relu = keras.layers.LeakyReLU()
        self.deconv_2 = keras.layers.Conv2DTranspose(3, 5, 2, output_padding=1, activation="sigmoid")

    def call(self, inputs):
        x = self.deconv_1(inputs)
        x = self.leaky_relu(x)
        return self.deconv_2(x)
