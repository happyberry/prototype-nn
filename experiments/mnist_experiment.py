import os

from experiments.base_experiment import BaseExperiment
import tensorflow as tf
import tensorflow.keras as keras

from src.data.dataset_loader import load_tf_data
from src.data.preprocessing import transform_tf, process, transform
from src.models.mnist_autoencoder import MnistAutoencoder
from src.models.proto_net import ProtoNet
from src.utils.image_utils import display_image


class MnistExperiment(BaseExperiment):

    def __init__(self, load_model: bool, batch_size=250, number_of_prototypes=15, number_of_epochs=100,
                 disable_r1=False, disable_r2=False, dataset_name="MNIST"):
        self.dataset_name = dataset_name
        self.number_of_classes = 10
        self.model = ProtoNet(MnistAutoencoder(), number_of_prototypes, self.number_of_classes)
        super().__init__(batch_size, number_of_prototypes, number_of_epochs, disable_r1, disable_r2, load_model)

    def init_datasets(self):
        train_dataset, val_dataset, test_dataset = load_tf_data(self.dataset_name, "train[:55000]"), \
                                                   load_tf_data(self.dataset_name, "train[55000:]"), \
                                                   load_tf_data(self.dataset_name, "test")
        train_ds = train_dataset.shuffle(1000).map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_dataset.map(self.preprocess_fn).batch(self.batch_size)
        test_ds = test_dataset.map(self.preprocess_fn).batch(self.batch_size)
        return train_ds, val_ds, test_ds

    @staticmethod
    def preprocess_fn(x, y):
        return tf.image.convert_image_dtype(x, tf.float32), y
        #return transform_tf(x, y)


def main():
    experiment = MnistExperiment(False, number_of_epochs=100, number_of_prototypes=15,
                                 disable_r1=False, disable_r2=False, dataset_name="FashionMNIST")
    experiment.run()
    experiment.decode_sample_images()
    img = next(iter(experiment.train_ds.take(1)))[0][:1]
    display_image(img[0])
    display_image(transform_tf(img, tf.constant([0]))[0])
    display_image(transform(img[0].numpy()))


if __name__ == "__main__":
    main()
