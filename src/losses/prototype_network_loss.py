import tensorflow as tf

from src.losses.autoencoder_loss import AutoencoderLoss
from src.losses.classification_loss import ClassificationLoss


class PrototypeNetworkLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_=0.05, lambda_1=0.05, lambda_2=0.05, name="prototype_network_loss"):
        super().__init__(name=name)
        self.weights = tf.constant([1.0, lambda_, lambda_1, lambda_2])
        self.lambda_ = lambda_
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.autoencoder_loss = AutoencoderLoss()
        self.classification_loss = ClassificationLoss()

    @tf.function
    def call(self, y_true, y_pred):
        label, image = y_true
        predicted_label, reconstruction, r1, r2 = y_pred
        classification_loss_value = self.classification_loss(label, predicted_label)
        autoencoder_loss_value = self.autoencoder_loss(image, reconstruction)
        values = tf.stack([classification_loss_value, autoencoder_loss_value, tf.convert_to_tensor(r1), tf.convert_to_tensor(r2)])
        return tf.tensordot(self.weights, values, axes=1)
