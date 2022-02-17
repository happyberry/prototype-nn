import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow.keras as keras

from src.data.dataset_loader import load_class_labels
from src.utils.image_utils import *


class BaseExperiment(ABC):
    model: keras.Model
    number_of_classes: int
    dataset_name: str

    def __init__(self, batch_size: int, number_of_prototypes: int, number_of_epochs: int, use_classic_model=False,
                 ablate=False, load_model=True):
        self.batch_size = batch_size
        self.number_of_prototypes = number_of_prototypes
        self.number_of_epochs = number_of_epochs
        self.train_ds, self.val_ds, self.test_ds = self.init_datasets()

        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.use_interpretable_model = not use_classic_model and not ablate
        self.load_model = load_model
        self.class_labels = load_class_labels(self.dataset_name)
        self.results_root = f"results/{self.dataset_name}/{self.number_of_epochs} epochs"
        Path(self.results_root + "/images").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def init_datasets(self):
        pass

    def train_model(self, show_epoch_time=True):
        for epoch in range(self.number_of_epochs):
            if show_epoch_time:
                if epoch > 0:
                    print(f"Time: {time.perf_counter() - start:.3f}s")
                start = time.perf_counter()
            losses = []
            print("\nEpoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                losses.append(self.model.train_step(x_batch_train, y_batch_train, self.train_acc_metric))
            epoch_loss = sum(losses) / len(losses)
            for x_batch_val, y_batch_val in self.val_ds:
                self.model.test_step(x_batch_val, y_batch_val, self.val_acc_metric)
            train_acc, val_acc = self.train_acc_metric.result(), self.val_acc_metric.result()
            print(f"Train loss: {epoch_loss:.4f} | Train accuracy: {100 * float(train_acc):.2f}% | Validation accuracy: {100 * float(val_acc):.2f}%")
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
            if epoch % 300 == 299 or epoch == self.number_of_epochs - 1:
                self.model.save_weights(f"{self.results_root}/model")

    def test_model(self):
        test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        for x_batch_val, y_batch_val in self.test_ds:
            self.model.test_step(x_batch_val, y_batch_val, test_acc_metric)
        print(f"Test accuracy: {100 * float(test_acc_metric.result()):.2f}%\n")

    def load_model_weights(self):
        self.model.load_weights(f"{self.results_root}/model")

    def display_results(self):
        self.decode_sample_images()
        desired_width = desired_height = int((self.model.prototype_layer.prototypes.shape[-1] / 10) ** 0.5)
        desired_shape = (self.number_of_prototypes, desired_height, desired_width, 10)
        decoded = self.model.autoencoder.decoder(tf.reshape(self.model.prototype_layer.prototypes, desired_shape))
        weights = self.model.classification_layer.weights[0].numpy()
        print(f"Prototype layer weights:\n{weights}\n")
        concatenated_images, title = concatenate_images(pad_images(decoded)), "Decoded learnt prototypes"
        display_image(concatenated_images, title)
        save_image(concatenated_images, f"{self.results_root}/images/{title}.png")
        print("Classes closest to prototypes:")
        print(f"{self.class_labels[tf.math.argmin(self.model.classification_layer.weights[0], axis=1).numpy()]}\n")
        sample_test_image = next(iter(self.test_ds.take(1)))[0][:1]
        title = "Sample test image"
        display_image(sample_test_image[0], title)
        save_image(sample_test_image[0].numpy(), f"{self.results_root}/images/{title}.png")
        print("Sample test image distance to each of the prototypes:")
        print(self.model.compute_distances_to_prototypes(self.model.autoencoder.encoder(sample_test_image[:1])[1])[0].numpy())

    def decode_sample_images(self, number_of_images=10):
        image_batch = next(iter(self.train_ds.take(1)))[0]
        decoded_batch = self.model.autoencoder(image_batch)
        images_to_concat = pad_images(image_batch[:number_of_images]), pad_images(decoded_batch[:number_of_images])
        long_image = concatenate_images(tf.concat(images_to_concat, axis=1))
        title = "Obrazy ze zbioru treningowego i ich rekonstrukcje z autoenkodera"
        display_image(long_image, title)
        save_image(long_image,  f"{self.results_root}/images/{title}.png")

    def show_sample_images(self, number_of_images=10):
        imgs = next(iter(self.train_ds.take(1)))[0]
        display_image(concatenate_images(pad_images(imgs[:number_of_images])), "Przykładowe obrazy ze zbioru treningowego")

    def run(self):
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        if self.load_model:
            try:
                self.load_model_weights()
            except tf.errors.NotFoundError:
                warnings.warn("File with model weights not found. Training from scratch...")
                self.train_model()
        else:
            self.train_model()
        self.test_model()
        if self.use_interpretable_model:
            self.display_results()
