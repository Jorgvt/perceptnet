import tensorflow as tf
from tensorflow.keras import layers

from .kernelidentity import KernelIdentity

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
                 kernel_initializer="identity",
                 data_format="channels_last",
                 **kwargs):

        super(GDN, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min+self.reparam_offset**2)**(1/2)
        self.apply_independently = apply_independently
        self.kernel_initializer = KernelIdentity(gain=gamma_init) if kernel_initializer=="identity" else kernel_initializer
        self.data_format = data_format
        
        self.alpha_init = alpha_init
        self.epsilon_init = epsilon_init
        self.alpha_trainable = alpha_trainable
        self.epsilon_trainable = epsilon_trainable        

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
                                  padding="valid", # We're using valid because the padding is done by hand with reflection
                                  strides=1,
                                  groups=self.groups,
                                  data_format=self.data_format,
                                  trainable=True,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_constraint=lambda x: tf.clip_by_value(x, 
                                                                       clip_value_min=self.reparam_offset,
                                                                       clip_value_max=tf.float32.max),
                                  bias_initializer="ones",
                                  bias_constraint=lambda x: tf.clip_by_value(x, 
                                                                      clip_value_min=self.beta_reparam,
                                                                      clip_value_max=tf.float32.max))
        self.conv.build(input_shape)

        # We have to define them here so that the names are properly set
        ## Actually, alpha should be a matrix as big as the kernel, and thus
        ## every element in X could be on a different power.
        self.alpha = self.add_weight(shape=(1),
                                     initializer=tf.keras.initializers.Constant(self.alpha_init),
                                     trainable=self.alpha_trainable,
                                     name='alpha')
        self.epsilon = self.add_weight(shape=(1),
                                       initializer=tf.keras.initializers.Constant(self.epsilon_init),
                                       trainable=self.epsilon_trainable,
                                       name='epsilon')
        ## Before que needed to define beta explicitly because we were using tf.nn.conv2d() and that doesnt allow
        ## the use of biases, but it's actually the bias. (The torch implementation uses it as the bias as well).
        # self.beta = self.add_weights(shape=n_channels,
        #                              initializer='ones',
        #                              constrain=lambda x: tf.clip_by_value(x, 
        #                                                                   clip_value_min=self.beta_reparam,
        #                                                                   clip_value_max=tf.float32.max),
        #                              name='beta')


    def call(self, X):
        """
        The PyTorch implementation works with inputs of shape:
        [batch_size, channels, height, width].
        We'll first copy it and then we'll try to change it.
        """
        ## We'll first pad the image by hand because doing it inside
        ## the layer only allows to pad with 0s and we want to pad with the
        ## reflection. As we're normalizing with the surrounding pixels,ç
        ## padding with 0s and padding with the reflection can have greatly
        ## different results at the edges.
        X_pad = tf.pad(X, 
                       mode = 'REFLECT',
                       paddings = tf.constant([[0, 0], # Batch dim
                                               [int((self.kernel_size-1)/2),
                                                int((self.kernel_size-1)/2)], 
                                               [int((self.kernel_size-1)/2), 
                                                int((self.kernel_size-1)/2)], 
                                               [0, 0]]))
        norm_pool = self.conv(tf.pow(X_pad, self.alpha))
        norm_pool = tf.pow(norm_pool, self.epsilon)

        return X / norm_pool

    def get_config(self):
        """
        Returns a dictionary used to initialize this layer. Is used when
        saving the layer or a model that contains it.
        """
        base_config = super(GDN, self).get_config()
        config = {'alpha':self.alpha,
                  'epsilon':self.epsilon}
        return dict(list(base_config.items()) + list(config.items()))