import tensorflow as tf

class KernelIdentity(tf.keras.initializers.Initializer):

    def __init__(self, gain):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        """
        shape has the form [Kx, Ky, Cin, Cout] disregarding data_format.
        """
        identity_matrix = tf.eye(shape[0])*self.gain
        identity_matrix = tf.expand_dims(identity_matrix, axis=-1)
        identity_matrix = tf.expand_dims(identity_matrix, axis=-1)
        identity_matrix = tf.repeat(identity_matrix, shape[2], axis=-2)
        identity_matrix = tf.repeat(identity_matrix, shape[3], axis=-1)
        return identity_matrix
    
    def get_config(self):
        return {'gain':self.gain}