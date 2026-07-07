import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

from .kernelidentity import KernelIdentity

@tf.keras.utils.register_keras_serializable(package="perceptnet")
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value_min=0.0):
        super(ClipConstraint, self).__init__()
        self.clip_value_min = float(clip_value_min)

    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=self.clip_value_min, clip_value_max=tf.float32.max)

    def get_config(self):
        return {'clip_value_min': self.clip_value_min}

@tf.keras.utils.register_keras_serializable(package="perceptnet")
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

        # Handle backward compatibility where 'alpha' and 'epsilon' were stored in the config
        kwargs.pop('alpha', None)
        kwargs.pop('epsilon', None)
        
        super(GDN, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min+self.reparam_offset**2)**(1/2)
        self.apply_independently = apply_independently
        
        if kernel_initializer == "identity":
            self.kernel_initializer = KernelIdentity(gain=gamma_init)
        else:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
            
        self.data_format = data_format
        
        self.alpha_init = alpha_init
        self.epsilon_init = epsilon_init
        self.alpha_trainable = alpha_trainable
        self.epsilon_trainable = epsilon_trainable        

    def build(self, input_shape):
        ## Extract the number of channels from the input shape
        ## according to the data_format
        if self.data_format=="channels_last":
            n_channels = input_shape[-1]
        elif self.data_format=="channels_first":
            # For channels_first (batch, channels, height, width), the channel index is 1.
            # But we keep input_shape[0] fallback if it's somehow unbatched.
            n_channels = input_shape[1] if len(input_shape) > 3 else input_shape[0]
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
                                  kernel_constraint=ClipConstraint(self.reparam_offset),
                                  bias_initializer="ones",
                                  bias_constraint=ClipConstraint(self.beta_reparam))
        self.conv.build(input_shape)

        self.alpha = self.add_weight(shape=(1,),
                                     initializer=tf.keras.initializers.Constant(self.alpha_init),
                                     trainable=self.alpha_trainable,
                                     name='alpha')
        self.epsilon = self.add_weight(shape=(1,),
                                       initializer=tf.keras.initializers.Constant(self.epsilon_init),
                                       trainable=self.epsilon_trainable,
                                       name='epsilon')

    def call(self, X):
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
        base_config = super(GDN, self).get_config()
        config = {
            'kernel_size': self.kernel_size,
            'gamma_init': self.gamma_init,
            'alpha_init': self.alpha_init,
            'epsilon_init': self.epsilon_init,
            'alpha_trainable': self.alpha_trainable,
            'epsilon_trainable': self.epsilon_trainable,
            'reparam_offset': self.reparam_offset,
            'beta_min': self.beta_min,
            'apply_independently': self.apply_independently,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer) if not isinstance(self.kernel_initializer, str) else self.kernel_initializer,
            'data_format': self.data_format,
        }
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="perceptnet")
class GDNCustom(layers.Layer):
    """GDN that takes as input a specific layer to use."""

    def __init__(self,
                 layer, # Layer to be used to extract the normalization.
                 alpha=2.0,
                 epsilon=0.5,
                 **kwargs):
        super(GDNCustom, self).__init__(**kwargs)
        self.layer = layer
        self.alpha_val = alpha
        self.epsilon_val = epsilon

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.alpha = self.add_weight(shape=(1,),
                                     initializer=tf.keras.initializers.Constant(self.alpha_val),
                                     trainable=False,
                                     name="alpha",
                                     dtype=tf.float32)
        self.epsilon = self.add_weight(shape=(1,),
                                       initializer=tf.keras.initializers.Constant(self.epsilon_val),
                                       trainable=False,
                                       name="epsilon",
                                       dtype=tf.float32)

    def call(self, X, training=False):
        norm = tf.math.pow(X, self.alpha)
        norm = self.layer(norm, training=training)
        norm = tf.clip_by_value(norm, clip_value_min=1e-5, clip_value_max=tf.reduce_max(norm))
        norm = tf.math.pow(norm, self.epsilon)
        return X / norm

    def get_config(self):
        base_config = super(GDNCustom, self).get_config()
        config = {
            'layer': tf.keras.layers.serialize(self.layer),
            'alpha': float(self.alpha.numpy()[0]) if hasattr(self, 'alpha') and hasattr(self.alpha, 'numpy') else float(self.alpha_val),
            'epsilon': float(self.epsilon.numpy()[0]) if hasattr(self, 'epsilon') and hasattr(self.epsilon, 'numpy') else float(self.epsilon_val),
        }
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if 'layer' in config:
            config['layer'] = tf.keras.layers.deserialize(config['layer'])
        return cls(**config)
