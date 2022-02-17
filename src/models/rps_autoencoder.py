import tensorflow as tf

from src.models.rps_decoder import RpsDecoder
from src.models.rps_encoder import RpsEncoder


class RpsAutoencoder(tf.keras.Model):
    def __init__(self):
        super(RpsAutoencoder, self).__init__()
        self.encoder = RpsEncoder()
        self.decoder = RpsDecoder()

    def call(self, inputs):
        encoded, _ = self.encoder(inputs)
        return self.decoder(encoded)
