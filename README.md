# Reimplementation of explainable prototype deep neural network

This repository contains reimplemented interpretable neural network proposed in
[paper](https://arxiv.org/pdf/1710.04806.pdf) by Oscar Li, Cynthia Rudin et al.

## Datasets

* Sadly, we didn't get any response from owners of `Cars` dataset, so (as planned) we switched
to [`Rock, Paper, Scissors`](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
dataset resized to 64x64 shape (to unify image shapes with `Cars`)
* All three datasets are available on [Tensorflow Datasets](https://www.tensorflow.org/datasets/).
Code used to download the datasets is available [here](https://github.com/happyberry/prototype-nn/blob/main/src/data/dataset_loader.py#L8)

## Authors' code
We executed the [code](https://github.com/OscarcarLi/PrototypeDL) shared by the paper authors on MNIST dataset.
As authors' code is compatible with Tensorflow 1.2, we used [docker image](https://hub.docker.com/layers/tensorflow/tensorflow/1.2.1-py3/images/sha256-04d55504c9985152ba62c7ce1e208a212a98bd0debbd34888b65e556f8a37b65?context=explore) to run the code.
To save time and energy, we ran the training for 100 epochs instead of 1500 proposed by authors as results were
satysfying by that point. Results of the experiment with authors' code are available [here](https://github.com/happyberry/prototype-nn/tree/main/authors_code_results).

## Implementation

* We implemented everything needed to conduct all the experiments described in paper, most notably including
[custom loss function](https://github.com/happyberry/prototype-nn/blob/main/src/losses/prototype_network_loss.py),
[data augmentation](https://github.com/happyberry/prototype-nn/blob/main/src/data/preprocessing.py) and 
[prototype layer](https://github.com/happyberry/prototype-nn/blob/main/src/layers/prototype_layer.py)
* Experiments which were originally conducted on `Cars` dataset were reproduced on FashionMNIST
* Comparison of our results and original ones are presented on slides which can be found [here](https://github.com/happyberry/prototype-nn/blob/main/presentation.pdf)