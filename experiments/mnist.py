import tensorflow as tf
import time
import tensorflow.keras as keras
from matplotlib import pyplot as plt

from src.data.dataset_loader import load_tf_data
from src.data.preprocessing import process, transform_tf
from src.losses.autoencoder_loss import AutoencoderLoss
from src.losses.classification_loss import ClassificationLoss
from src.losses.prototype_network_loss import PrototypeNetworkLoss
from src.models.proto_net import MnistProtoNet


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits, reconstruction, r1, r2 = model(x, training=True)
        loss_value = prototype_loss((y, x), (logits, reconstruction, r1, r2))
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y, metric):
    logits, _, _, _ = model(x, training=False)
    metric.update_state(y, logits)


batch_size = 250
number_of_prototypes = 15
number_of_epochs = 100
dataset_name = "MNIST"

train_dataset, val_dataset, test_dataset = load_tf_data(dataset_name, "train[:55000]"),\
                                           load_tf_data(dataset_name, "train[55000:]"), \
                                           load_tf_data(dataset_name, "test")
train_ds = train_dataset.shuffle(1000).map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)) \
    .batch(batch_size)
val_ds = val_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)).batch(batch_size)
test_ds = test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)).batch(batch_size)

model = MnistProtoNet(number_of_prototypes)

classification_loss_fn = ClassificationLoss()
autoencoder_loss_fn = AutoencoderLoss()
prototype_loss = PrototypeNetworkLoss()
optimizer = keras.optimizers.Adam(learning_rate=0.002)
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
for epoch in range(number_of_epochs):
    if epoch > 0:
        print(time.time() - start)
    start = time.time()
    losses = []
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
        losses.append(train_step(x_batch_train, y_batch_train))
    epoch_loss = sum(losses) / len(losses)
    for x_batch_val, y_batch_val in val_ds:
        test_step(x_batch_val, y_batch_val, val_acc_metric)
    train_acc, val_acc = train_acc_metric.result(), val_acc_metric.result()
    print(f"Loss w epoce: {epoch_loss}\n Train acc:{float(train_acc)}\nVal acc:{float(val_acc)}")
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
for x_batch_val, y_batch_val in test_ds:
    test_step(x_batch_val, y_batch_val, test_acc_metric)
print(f"Dokładność na zbiorze testowym:{float(test_acc_metric.result())}")

decoded = model.decoder(tf.reshape(model.prototype_layer.prototypes, (number_of_prototypes, 2, 2, 10)), training=False)
weights = model.classification_layer.weights[0].numpy()
for i, image in enumerate(decoded.numpy()):
    print(weights[i])
    plt.imshow(image, cmap='gray')
    plt.show()
print(tf.math.argmin(model.classification_layer.weights[0], axis=1))
