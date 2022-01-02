import tensorflow as tf
import tensorflow.keras as keras


class MnistDecoder(tf.keras.Model):

    def __init__(self):
        super(MnistDecoder, self).__init__()
        self.deconv_1 = keras.layers.Conv2DTranspose(10, 3, 2, padding='same', activation="sigmoid")
        self.deconv_2 = keras.layers.Conv2DTranspose(32, 3, 2, padding='same', output_padding=0, activation="sigmoid")
        self.deconv_3 = keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation="sigmoid")
        self.deconv_4 = keras.layers.Conv2DTranspose(1, 3, 2, padding='same', activation="sigmoid")

    def call(self, inputs):
        x = self.deconv_1(inputs)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        return self.deconv_4(x)
