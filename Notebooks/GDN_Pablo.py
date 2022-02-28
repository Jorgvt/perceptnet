from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg

#GDN layer
class GDN(Layer):
    def __init__(self, 
                 filter_shape = (3,3), 
                 **kwargs):
      
        self.filter_shape = filter_shape

        super(GDN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name = 'beta', 
                                    shape = (input_shape.as_list()[-1]),
                                    initializer = tf.keras.initializers.constant(1.0),
                                    trainable = True,
                                    constraint = lambda x: tf.clip_by_value(x, 1e-15, np.inf))
        
        self.alpha = self.add_weight(name = 'alpha', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(1.0),
                                     trainable = False,
                                     constraint = NonNeg())

        self.epsilon = self.add_weight(name = 'epsilon', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(1.0),
                                     trainable = False,
                                     constraint = NonNeg())
        
        self.gamma = self.add_weight(name = 'gamma', 
                                     shape = (self.filter_shape[0], self.filter_shape[1], input_shape.as_list()[-1], input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.Zeros,
                                     trainable = True,
                                     constraint = NonNeg())
        
        
        super(GDN, self).build(input_shape)

    def call(self, x):
        norm_conv2 = tf.nn.convolution(tf.abs(x)**self.alpha,
                                      self.gamma,
                                      strides = (1, 1),
                                      padding = "SAME",
                                      data_format = "NHWC")

        norm_conv = self.beta + norm_conv2
        norm_conv = norm_conv**self.epsilon
        return x / norm_conv
        
    def compute_output_shape(self, input_shape):
        return (input_shape, self.output_dim)
