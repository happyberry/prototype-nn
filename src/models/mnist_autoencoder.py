import tensorflow as tf

from src.models.mnist_decoder import MnistDecoder
from src.models.mnist_encoder import MnistEncoder


class MnistAutoencoder(tf.keras.Model):
    def __init__(self):
        super(MnistAutoencoder, self).__init__()
        self.encoder = MnistEncoder()
        self.decoder = MnistDecoder()

    def call(self, inputs):
        encoded, _ = self.encoder(inputs)
        return self.decoder(encoded)
