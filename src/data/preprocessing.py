import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.data import Dataset

ALPHA = 20
SIGMA = 4


def process(image, label):
    return tf.numpy_function(transform, [image], tf.uint8), label


def transform(image):
    
    height, width, _ = image.shape
    # print(image.shape)
    image = image.reshape((28,28))
    # print(image.shape)
    # return image
    x, y = np.mgrid[0:height, 0:width]

    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), SIGMA, mode='constant') * ALPHA
    indices = x + dx, y + dy
    # print(image.shape, len(indices[0]))
    image = map_coordinates(image, indices, order=1)
    image = image.reshape((28,28,1))
    return tf.convert_to_tensor(image)


def transform_ds(ds: Dataset):
    return ds.map(process)
