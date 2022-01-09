import tensorflow as tf
import tensorflow.keras as keras

from src.layers.prototype_layer import PrototypeLayer
from src.losses.prototype_network_loss import PrototypeNetworkLoss


class ProtoNet(keras.Model):
    def __init__(self, autoencoder, number_of_prototypes, number_of_classes, disable_r1, disable_r2, ablated=False):
        super(ProtoNet, self).__init__()
        self.autoencoder = autoencoder
        if ablated:
            self.dense = keras.layers.Dense(number_of_prototypes)
        else:
            self.prototype_layer = PrototypeLayer(number_of_prototypes)
        self.classification_layer = keras.layers.Dense(number_of_classes)

        self.loss_fn = PrototypeNetworkLoss(disable_r1=disable_r1, disable_r2=disable_r2)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.002)
        self.ablated = ablated

    def call(self, inputs):
        encoded, flattened_encoded = self.autoencoder.encoder(inputs)
        decoded = self.autoencoder.decoder(encoded)
        if self.ablated:
            return self.classification_layer(self.dense(flattened_encoded)), decoded, tf.constant(0.), tf.constant(0.)
        prototype_distances, r1, r2 = self.compute_distances_to_prototypes(flattened_encoded)
        return self.classification_layer(prototype_distances), decoded, r1, r2

    def compute_distances_to_prototypes(self, flattened_encoded):
        return self.prototype_layer(flattened_encoded)

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor, metric: keras.metrics.Metric):
        with tf.GradientTape() as tape:
            logits, reconstruction, r1, r2 = self(x, training=True)
            loss_value = self.loss_fn((y, x), (logits, reconstruction, r1, r2))
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x: tf.Tensor, y: tf.Tensor, metric: keras.metrics.Metric):
        logits, _, _, _ = self(x, training=False)
        metric.update_state(y, logits)
