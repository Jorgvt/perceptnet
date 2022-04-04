import tensorflow as tf
from tensorflow.keras import layers

from GDN_Jorge import GDN as GDNJ
from pearson_loss import PearsonCorrelation

class PerceptNet(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False):
        super(PerceptNet, self).__init__()
        self.gdn1 = GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer)
        self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn2 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn3 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same')
        self.gdn4 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)

    def call(self, X):
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output)
        output = self.gdn4(output)
        return output

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self(img)
            features_distorted = self(dist_img)
            l2 = (features_original-features_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'pearson':loss}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img)
        features_distorted = self(dist_img)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        return {'pearson':loss}

class PerceptNetRegressor(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False):
        super(PerceptNetRegressor, self).__init__()
        self.gdn1 = GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer)
        self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn2 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn3 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same')
        self.gdn4 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.flatten = layers.Flatten()
        self.regressor = layers.Dense(1)
        self.correlation_loss = PearsonCorrelation()

    def extract_features(self, X):
        features = self.gdn1(X)
        features = self.conv1(features)
        features = self.undersampling1(features)
        features = self.gdn2(features)
        features = self.conv2(features)
        features = self.undersampling2(features)
        features = self.gdn3(features)
        features = self.conv3(features)
        features = self.gdn4(features)
        return features

    def call(self, X):
        X1, X2 = X
        features1 = self.extract_features(X1)
        features2 = self.extract_features(X2)
        features = layers.concatenate([features1, features2])
        output = self.flatten(features)
        output = self.regressor(output)
        return output

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        ## Forward pass and backprop
        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self.extract_features(img)
            features_distorted = self.extract_features(dist_img)
            features = layers.concatenate([features_original, features_distorted])
            features = self.flatten(features)
            mos_pred = self.regressor(features)
            loss = self.compiled_loss(mos, mos_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ## Evaluating the Pearson Correlation
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        correlation = self.correlation_loss(mos, l2)

        return {'loss':loss, 'pearson':correlation}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self.extract_features(img)
        features_distorted = self.extract_features(dist_img)
        features = layers.concatenate([features_original, features_distorted])
        features = self.flatten(features)
        mos_pred = self.regressor(features)
        loss = self.compiled_loss(mos, mos_pred)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        correlation = self.correlation_loss(mos, l2)
        return {'loss':loss, 'pearson':correlation}