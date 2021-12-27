import tensorflow as tf


class AutoencoderLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.math.square(tf.norm(y_true - y_pred))