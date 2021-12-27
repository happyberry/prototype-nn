import tensorflow as tf


class ClassificationLoss(tf.keras.losses.Loss):
    def __init__(self, name="classification_loss"):
        super().__init__(name=name)
        self.crossEntropyLoss = tf.keras.losses.CategoricalCrossEntropy(from_logits=True)

    def call(self, y_true, y_pred):
        return self.crossEntropyLoss(y_true, y_pred)