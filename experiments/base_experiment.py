import time
import warnings
from abc import ABC, abstractmethod

import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow.keras as keras

from src.losses.prototype_network_loss import PrototypeNetworkLoss
from src.utils.image_utils import *


class BaseExperiment(ABC):
    model: keras.Model
    number_of_classes: int
    dataset_name: str

    def __init__(self, batch_size: int, number_of_prototypes: int, number_of_epochs: int,
                 disable_r1: bool, disable_r2: bool, load_model=True):
        self.batch_size = batch_size
        self.number_of_prototypes = number_of_prototypes
        self.number_of_epochs = number_of_epochs
        self.train_ds, self.val_ds, self.test_ds = self.init_datasets()

        self.prototype_loss = PrototypeNetworkLoss(disable_r1=disable_r1, disable_r2=disable_r2)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.002)

        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.load_model = load_model

    @abstractmethod
    def init_datasets(self):
        pass

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor):
        with tf.GradientTape() as tape:
            logits, reconstruction, r1, r2 = self.model(x, training=True)
            loss_value = self.prototype_loss((y, x), (logits, reconstruction, r1, r2))
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x: tf.Tensor, y: tf.Tensor, metric: keras.metrics.Metric):
        logits, _, _, _ = self.model(x, training=False)
        metric.update_state(y, logits)

    def train_model(self, show_epoch_time=True):
        for epoch in range(self.number_of_epochs):
            if show_epoch_time:
                if epoch > 0:
                    print(f"Time: {time.perf_counter() - start:.3f}s")
                start = time.perf_counter()
            losses = []
            print("\nEpoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                losses.append(self.train_step(x_batch_train, y_batch_train))
            epoch_loss = sum(losses) / len(losses)
            for x_batch_val, y_batch_val in self.val_ds:
                self.test_step(x_batch_val, y_batch_val, self.val_acc_metric)
            train_acc, val_acc = self.train_acc_metric.result(), self.val_acc_metric.result()
            print(f"Train loss: {epoch_loss:.4f} | Train accuracy: {100 * float(train_acc):.2f}% | Validation accuracy: {100 * float(val_acc):.2f}%")
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
            if epoch % 100 == 99:
                self.model.save_weights(f"results/{self.dataset_name}/model")

    def test_model(self):
        test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        for x_batch_val, y_batch_val in self.test_ds:
            self.test_step(x_batch_val, y_batch_val, test_acc_metric)
        print(f"Test accuracy: {100 * float(test_acc_metric.result()):.2f}%")

    def load_model_weights(self):
        self.model.load_weights(f"results/{self.dataset_name}/model")

    def display_results(self):
        desired_width = desired_height = int((self.model.prototype_layer.prototypes.shape[-1] / 10) ** 0.5)
        desired_shape = (self.number_of_prototypes, desired_height, desired_width, 10)
        decoded = self.model.autoencoder.decoder(tf.reshape(self.model.prototype_layer.prototypes, desired_shape))
        weights = self.model.classification_layer.weights[0].numpy()
        print(weights)
        display_image(concatenate_images(pad_images(decoded)), "Zdekodowane prototypy wyuczone przez model")
        print(tf.math.argmin(self.model.classification_layer.weights[0], axis=1))

    def decode_sample_images(self, number_of_images=10):
        image_batch = next(iter(self.train_ds.take(1)))[0]
        decoded_batch = self.model.autoencoder(image_batch)
        images_to_concat = pad_images(image_batch[:number_of_images]), pad_images(decoded_batch[:number_of_images])
        long_image = concatenate_images(tf.concat(images_to_concat, axis=1))
        display_image(long_image, "Obrazy ze zbioru treningowego i ich rekonstrukcje z autoenkodera")

    def show_sample_images(self, number_of_images=10):
        imgs = next(iter(self.train_ds.take(1)))[0]
        display_image(concatenate_images(pad_images(imgs[:number_of_images])), "Przykładowe obrazy ze zbioru treningowego")

    def run(self):
        if self.load_model:
            try:
                self.load_model_weights()
            except tf.errors.NotFoundError:
                warnings.warn("File with model weights not found. Training from scratch...")
                self.train_model()
        else:
            self.train_model()
        self.test_model()
        self.display_results()
