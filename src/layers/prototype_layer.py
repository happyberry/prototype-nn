import tensorflow as tf


class PrototypeLayer(tf.keras.layers.Layer):
    def __init__(self, number_of_prototypes):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = number_of_prototypes

    def build(self, input_shape):
        self.prototypes = self.add_weight("prototypes", shape=[self.num_prototypes, int(input_shape[-1])],
                                          initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0))

    def call(self, inputs):
        prototypes = tf.expand_dims(self.prototypes, 0)
        ins = tf.expand_dims(inputs, 1)
        kek = prototypes - ins
        distance_matrix = tf.math.square(tf.norm(kek, axis=-1)) # axis 0 - batch, axis 1 - prototypes
        r1 = tf.math.reduce_mean(tf.math.reduce_min(distance_matrix, axis=0))
        r2 = tf.math.reduce_mean(tf.math.reduce_min(distance_matrix, axis=1))
        return distance_matrix, r1, r2
