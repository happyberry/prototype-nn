import tensorflow as tf


class PrototypeNetworkLoss(tf.keras.losses.Loss):
    def __init__(self, autoencoder_loss_value, r1_term, r2_term, lambda_=0.05, lambda_1=0.05, lambda_2=0.05, name="prototype_network_loss"):
        super().__init__(name=name)
        self.crossEntropyLoss = tf.keras.losses.CategoricalCrossEntropy(from_logits=False)
        self.r1 = r1_term
        self.r2 = r2_term
        self.weights = tf.constant([1.0, lambda_, lambda_1, lambda_2])
        self.lambda_ = lambda_
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.autoencoder_loss_value = autoencoder_loss_value
        self.classification_loss = ClassificationLoss()

    def call(self, y_true, y_pred):
        classification_loss_value = self.classification_loss(y_true, y_pred).item()
        values = tf.constant([classification_loss_value, self.autoencoder_loss_value, self.r1, self.r2])
        return tf.tensordot(self.weights, values, axes=1)