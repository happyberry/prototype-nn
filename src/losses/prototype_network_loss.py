from typing import Tuple

import tensorflow as tf

from src.losses.autoencoder_loss import AutoencoderLoss
from src.losses.classification_loss import ClassificationLoss


class PrototypeNetworkLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_=0.05, lambda_1=0.05, lambda_2=0.05, disable_r1=False, disable_r2=False, name="prototype_network_loss"):
        super().__init__(name=name)
        self.disable_r1 = disable_r1
        self.disable_r2 = disable_r2
        self.lambda_ = lambda_
        self.lambda_1 = lambda_1 if not disable_r1 else 0
        self.lambda_2 = lambda_2 if not disable_r2 else 0
        self.weights = tf.constant([1.0, self.lambda_, self.lambda_1, self.lambda_2])
        self.autoencoder_loss = AutoencoderLoss()
        self.classification_loss = ClassificationLoss()

    @tf.function
    def call(self, y_true: Tuple[tf.Tensor, tf.Tensor], y_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
        label, image = y_true
        predicted_label, reconstruction, r1, r2 = y_pred
        classification_loss_value = self.classification_loss(label, predicted_label)
        autoencoder_loss_value = self.autoencoder_loss(image, reconstruction)
        values = tf.stack([classification_loss_value, autoencoder_loss_value, tf.convert_to_tensor(r1), tf.convert_to_tensor(r2)])
        return tf.tensordot(self.weights, values, axes=1)
