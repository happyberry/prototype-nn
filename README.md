# Reimplementation of explainable prototype deep neural network

## Progress

### Minimal requirements

* Sadly, we didn't get any response from owners of `Cars` dataset, so we switched to [`Rock, Paper, Scissors`](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) dataset resized to 64x64 shape (to make it similar to `Cars`)
* All three datasets are available on [Tensorflow Datasets](https://www.tensorflow.org/datasets/). Code used to download the datasets is available [here](https://github.com/happyberry/prototype-nn/blob/main/src/data/dataset_loader.py#L8). 
* Also, we ran the code shared by the paper authors on MNIST dataset. As authors' code is compatible with Tensorflow 1.2, we used [docker image](https://hub.docker.com/layers/tensorflow/tensorflow/1.2.1-py3/images/sha256-04d55504c9985152ba62c7ce1e208a212a98bd0debbd34888b65e556f8a37b65?context=explore) to run the code. To save time and energy, we ran the code for 100 epochs instead of 1500 proposed by authors. Results of the experiment with authors' code are available [here](https://github.com/happyberry/prototype-nn/tree/main/authors_code_results).

### Implementation progress

* We implemented everything needed to conduct MNIST/FashionMNIST experiments, most notably including [custom loss function](https://github.com/happyberry/prototype-nn/blob/main/src/losses/prototype_network_loss.py), [data augmentation](https://github.com/happyberry/prototype-nn/blob/main/src/data/preprocessing.py) and [prototype layer](https://github.com/happyberry/prototype-nn/blob/main/src/layers/prototype_layer.py).
* Code structure is similar to the proposed one, so hopefully everything interesting will be easy to find.
* Now we are working on all the visualizations, extensions needed for ablation studies and autoencoder architecture for `Rock, Paper, Scissors` dataset.
