import tensorflow as tf
import tensorflow.keras as keras

from src.layers.prototype_layer import PrototypeLayer


class ProtoNet(tf.keras.Model):
    def __init__(self, autoencoder, number_of_prototypes, number_of_classes):
        super(ProtoNet, self).__init__()
        self.autoencoder = autoencoder
        self.prototype_layer = PrototypeLayer(number_of_prototypes)
        self.classification_layer = keras.layers.Dense(number_of_classes)

    def call(self, inputs):
        encoded, flattened_encoded = self.autoencoder.encoder(inputs)
        decoded = self.autoencoder.decoder(encoded)
        prototype_distances, r1, r2 = self.prototype_layer(flattened_encoded)
        return self.classification_layer(prototype_distances), decoded, r1, r2
