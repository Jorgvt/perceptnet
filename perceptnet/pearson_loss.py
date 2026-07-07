import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="perceptnet")
class PearsonCorrelation(tf.keras.losses.Loss):
    """
    Loss used to train PerceptNet. Is calculated as the 
    Pearson Correlation Coefficient for a sample.
    """

    def __init__(self, name="pearson", **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        num = y_true-y_true_mean
        num *= y_pred-y_pred_mean
        num = tf.reduce_sum(num)
        denom = tf.sqrt(tf.reduce_sum((y_true-y_true_mean)**2))
        denom *= tf.sqrt(tf.reduce_sum((y_pred-y_pred_mean)**2))
        return num/denom

    def get_config(self):
        config = {'name': self.name}
        base_config = super(PearsonCorrelation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))