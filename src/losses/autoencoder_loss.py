import tensorflow as tf


class AutoencoderLoss(tf.keras.losses.Loss):
    def __init__(self, name="autoencoder_loss"):
        super().__init__(name=name)
        self.mseLoss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        return self.mseLoss(y_true, y_pred)