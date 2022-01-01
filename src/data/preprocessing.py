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
    image = image[:,:,np.newaxis]
    return tf.convert_to_tensor(image)


def transform_ds(ds: Dataset):
    return ds.map(process)
