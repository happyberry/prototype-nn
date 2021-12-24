import tensorflow as tf

class PrototypeLayer(tf.keras.layers.Layer):
    def __init__(self, number_of_prototypes):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = number_of_prototypes

    def build(self, input_shape):
        self.prototypes = self.add_weight("prototypes", shape=[self.num_prototypes, int(input_shape[-1])],
                                          initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0))

    def call(self, inputs):
        return tf.math.square(tf.norm(self.prototypes - inputs, axis=-1))  # implicit broadcast of 'inputs' in tf.norm