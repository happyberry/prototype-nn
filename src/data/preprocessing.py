import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.data import Dataset
import tensorflow_addons as tfa

ALPHA = 20
SIGMA = 4


def process(image, label):
    return tf.numpy_function(transform, [image], tf.double), label


def transform(image: np.ndarray):
    
    height, width, _ = image.shape
    image = image.squeeze()
    x, y = np.mgrid[0:height, 0:width]

    # TODO: add posibility of setting random state
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
    indices = x + dx, y + dy
    image = map_coordinates(image, indices, order=1)
    image = image[:, :, np.newaxis] / 255
    return tf.convert_to_tensor(image)


def transform_ds(ds: Dataset):
    return ds.map(process)


@tf.function
def transform_tf(image, label):  # RIP, scipy is faster
    image = tf.squeeze(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    x, y = tf.meshgrid(range(image.shape[1]), range(image.shape[0]))
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    filter_width = 2 * int(4 * SIGMA + 0.5) + 1  # formula from scipy implementation
    dx = tfa.image.gaussian_filter2d(tf.random.uniform(image.shape, -1, 1), filter_shape=filter_width, sigma=SIGMA, padding='CONSTANT') * ALPHA
    dy = tfa.image.gaussian_filter2d(tf.random.uniform(image.shape, -1, 1), filter_shape=filter_width, sigma=SIGMA, padding='CONSTANT') * ALPHA
    #dxn, dyn = dx.numpy(), dy.numpy()
    x, y = x + dx, y + dy
    indices = tf.stack([x, y], axis=-1)
    reshaped_image = tf.reshape(image, (1, *image.shape, 1))
    deformed_image = tfa.image.resampler(reshaped_image, tf.expand_dims(indices, axis=0))
    return tf.squeeze(deformed_image, axis=0), label


