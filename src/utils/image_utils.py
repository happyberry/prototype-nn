import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def display_image(image: np.array, title=""):
    plt.figure(figsize=(7, 3))
    if image.shape[-1] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def pad_images(images: tf.Tensor, pad_size=2):
    return tf.pad(images, tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]), constant_values=1.)


def concatenate_images(images: tf.Tensor):
    return np.concatenate(images.numpy(), axis=1)


def save_image(image: np.array, path: str):
    if image.shape[-1] == 1:
        plt.imsave(path, np.concatenate([image, image, image], axis=2))
    else:
        plt.imsave(path, image)
