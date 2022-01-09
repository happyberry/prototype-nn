from experiments.base_experiment import BaseExperiment
from src.data.dataset_loader import load_tf_data
from src.data.preprocessing import preprocess_rps_image, transform_tf
from src.models.proto_net import ProtoNet
from src.models.rps_autoencoder import RpsAutoencoder
from src.utils.image_utils import display_image


class RpsExperiment(BaseExperiment):

    def __init__(self, load_model: bool, batch_size=16, number_of_prototypes=3, number_of_epochs=100,
                 disable_r1=False, disable_r2=False, dataset_name="RockPaperScissors"):
        self.dataset_name = dataset_name
        self.number_of_classes = 3
        self.model = ProtoNet(RpsAutoencoder(), number_of_prototypes, self.number_of_classes, disable_r1, disable_r2)
        super().__init__(batch_size, number_of_prototypes, number_of_epochs, load_model=load_model)

    def init_datasets(self):
        train_dataset, val_dataset, test_dataset = load_tf_data(self.dataset_name, "train[:2300]"), \
                                                    load_tf_data(self.dataset_name, "train[2300:]"), \
                                                    load_tf_data(self.dataset_name, "test")
        train_ds = train_dataset.shuffle(1000).map(preprocess_rps_image).batch(self.batch_size)
        val_ds = val_dataset.map(preprocess_rps_image).batch(self.batch_size)
        test_ds = test_dataset.map(preprocess_rps_image).batch(self.batch_size)
        return train_ds, val_ds, test_ds


def main():
    experiment = RpsExperiment(True, batch_size=64, number_of_epochs=20, number_of_prototypes=5,
                                 disable_r1=False, disable_r2=False)
    experiment.run()
    sample = next(iter(experiment.train_ds.take(1)))[:]
    display_image(sample[0][0].numpy())
    display_image(transform_tf(sample[0][0], sample[1][0])[0].numpy())


if __name__ == "__main__":
    main()
