import tensorflow as tf
import tensorflow.keras as keras

from src.losses.classification_loss import ClassificationLoss


class MnistClassicCNN(tf.keras.Model):

    def __init__(self):
        super(MnistClassicCNN, self).__init__()
        self.conv_1 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="relu")
        self.conv_2 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="relu")
        self.conv_3 = keras.layers.Conv2D(32, 3, 2, padding='same', activation="relu")
        self.conv_4 = keras.layers.Conv2D(10, 3, 2, padding='same', activation="relu")
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(15, activation="relu")
        self.dense_2 = keras.layers.Dense(10)

        self.loss_fn = ClassificationLoss()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.002)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor, metric: keras.metrics.Metric):
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x: tf.Tensor, y: tf.Tensor, metric: keras.metrics.Metric):
        logits = self(x, training=False)
        metric.update_state(y, logits)
