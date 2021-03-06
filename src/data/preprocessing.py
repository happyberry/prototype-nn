import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.data import Dataset
import tensorflow_addons as tfa

from src.utils.image_utils import display_image

ALPHA = 20
SIGMA = 4


def process(image, label):
    return tf.numpy_function(transform, [image], tf.float32), label


def transform(image_batch: np.ndarray):
    
    height, width, _ = image_batch.shape[1:]
    x, y = np.mgrid[0:height, 0:width]

    random_state = np.random.RandomState(None)
    i = 0
    for img in image_batch:
        img = img.squeeze()
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
        indices = x + dx, y + dy
        img = map_coordinates(img, indices, order=1)
        image_batch[i] = img[:, :, np.newaxis]
        i += 1
    image_batch = image_batch / 255
    return tf.convert_to_tensor(image_batch, dtype=tf.float32)


def transform_ds(ds: Dataset):
    return ds.map(process)


@tf.function
def transform_tf(image, label):  # RIP, scipy is faster
    if len(image.shape) > 3 and image.shape[0] == 1:
        image = tf.squeeze(image, axis=0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    x, y = tf.meshgrid(range(image.shape[1]), range(image.shape[0]))
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    filter_width = 2 * int(4 * SIGMA + 0.5) + 1  # formula from scipy implementation
    dx = tfa.image.gaussian_filter2d(tf.random.uniform(image.shape[:2], -1, 1), filter_shape=filter_width, sigma=SIGMA, padding='CONSTANT') * ALPHA
    dy = tfa.image.gaussian_filter2d(tf.random.uniform(image.shape[:2], -1, 1), filter_shape=filter_width, sigma=SIGMA, padding='CONSTANT') * ALPHA
    #dxn, dyn = dx.numpy(), dy.numpy()
    x, y = x + dx, y + dy
    indices = tf.stack([x, y], axis=-1)
    reshaped_image = tf.expand_dims(image, axis=0)
    deformed_image = tfa.image.resampler(reshaped_image, tf.expand_dims(indices, axis=0))
    return tf.squeeze(deformed_image, axis=0), label


@tf.function
def preprocess_rps_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.numpy_function(transform, [image], tf.float32)
    resized_image = tf.image.resize(image, tf.constant([64, 64]))
    #resized_image, label = transform_tf(resized_image, label)
    return resized_image, label


