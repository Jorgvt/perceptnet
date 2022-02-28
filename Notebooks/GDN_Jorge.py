import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

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

class GDN(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size=3,
                 gamma_init=.1,
                 alpha_init=2,
                 epsilon_init=1/2,
                 alpha_trainable=False,
                 epsilon_trainable=False,
                 reparam_offset=2**(-18),
                 beta_min=1e-6,
                 apply_independently=False,
                 data_format="channels_last"):

        super(GDN, self).__init__()
        self.kernel_size = kernel_size
        # self.n_channels = n_channels
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min+self.reparam_offset**2)**(1/2)
        self.apply_independently = apply_independently
        self.data_format = data_format
        
        ## Trainable parameters
        self.alpha = self.add_weight(shape=(1),
                                     initializer=tf.keras.initializers.Constant(alpha_init),
                                     trainable=alpha_trainable,
                                     name='alpha')
        self.epsilon = self.add_weight(shape=(1),
                                       initializer=tf.keras.initializers.Constant(epsilon_init),
                                       trainable=epsilon_trainable,
                                       name='epsilon')

    def build(self, input_shape):
        ## Extract the number of channels from the input shape
        ## according to the data_format
        n_channels = input_shape[-1] if self.data_format=="channels_last" else input_shape[0]

        if self.data_format=="channels_last":
            n_channels = input_shape[-1]
        elif self.data_format=="channels_first":
            n_channels = input_shape[0]
        else:
            raise ValueError("data_format not supported")

        if self.apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1

        self.conv = layers.Conv2D(filters=n_channels,
                                  kernel_size=self.kernel_size,
                                  padding="same",
                                  strides=1,
                                  groups=self.groups,
                                  data_format=self.data_format,
                                  trainable=True,
                                  kernel_initializer=KernelIdentity(gain=self.gamma_init),
                                  kernel_constraint=lambda x: tf.clip_by_value(x, 
                                                                       clip_value_min=self.reparam_offset,
                                                                       clip_value_max=tf.float32.max),
                                  bias_initializer="ones",
                                  bias_constraint=lambda x: tf.clip_by_value(x, 
                                                                      clip_value_min=self.beta_reparam,
                                                                      clip_value_max=tf.float32.max))
        self.conv.build(input_shape)


    def call(self, X):
        """
        The PyTorch implementation works with inputs of shape:
        [batch_size, channels, height, width].
        We'll first copy it and then we'll try to change it.
        """
        norm_pool = self.conv(tf.pow(X, self.alpha))
        norm_pool = tf.pow(norm_pool, self.epsilon)

        return X / norm_pool