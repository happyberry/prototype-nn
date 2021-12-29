import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
import matplotlib.pyplot as plt

from src.data.preprocessing import transform_ds


def load_tf_data(dataset_name: str, split_type: str) -> PrefetchDataset:
    """After load it is possible to use returned object with .batch(batch_size)
        TODO: change to pydoc or delete this comment"""
    return tfds.load(name=dataset_name, split=split_type, as_supervised=True)


dataset = load_tf_data("mnist", "test")
for image, label in dataset.take(1):
    print(image.shape, label)
dataset = dataset.apply(transform_ds)
plt.figure(figsize=(10, 10))
for image, label in dataset.take(1): 
    print(image.shape, label)

