import tensorflow as tf
import tensorflow.keras as keras

from src.layers.prototype_layer import PrototypeLayer
from src.models.mnist_decoder import MnistDecoder
from src.models.mnist_encoder import MnistEncoder


class MnistProtoNet(tf.keras.Model):
    def __init__(self, number_of_prototypes):
        super(MnistProtoNet, self).__init__()
        self.encoder = MnistEncoder()
        self.decoder = MnistDecoder()
        self.prototype_layer = PrototypeLayer(number_of_prototypes)
        self.classification_layer = keras.layers.Dense(10)

    def call(self, inputs):
        #return self.classification_layer(self.encoder(inputs)[1])
        encoded, flattened_encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        prototype_distances, r1, r2 = self.prototype_layer(flattened_encoded)
        return self.classification_layer(prototype_distances), decoded, r1, r2
