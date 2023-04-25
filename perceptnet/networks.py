from einops import rearrange, repeat, reduce
from einops.layers.tensorflow import Rearrange

import tensorflow as tf
from tensorflow.keras import layers

from perceptnet.GDN_Jorge import GDN as GDNJ
from perceptnet.GDN_Jorge import GDNCustom
from perceptnet.pearson_loss import PearsonCorrelation

from flayers.layers import RandomGabor, PseudoRandomGabor
import flayers.experimental.layers as eflayers
from flayers.center_surround import RandomGaussian

class BasePercetNet(tf.keras.Model):
    def __init__(self, feature_extractor, **kwargs):
        super(BasePercetNet, self).__init__(**kwargs)
        self.feature_extractor = feature_extractor

    @property
    def layers(self):
        return self.feature_extractor.layers
    
    def call(self, X, training=False):
        return self.feature_extractor(X, training)

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self(img, training=True)
            features_distorted = self(dist_img, training=True)
            l2 = (features_original-features_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img, training=False)
        features_distorted = self(dist_img, training=False)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}

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

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img)
        features_distorted = self(dist_img)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}

class PerceptNetSeq(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False):
        super(PerceptNetSeq, self).__init__()
        self.model = tf.keras.Sequential([
            GDNJ(kernel_size=1, apply_independently=True, kernel_initializer=kernel_initializer),
            layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
            layers.MaxPool2D(2),
            GDNJ(kernel_size=1, kernel_initializer=kernel_initializer),
            layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
            layers.MaxPool2D(2),
            GDNJ(kernel_size=1, kernel_initializer=kernel_initializer),
            layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
            GDNJ(kernel_size=1, kernel_initializer=kernel_initializer)
        ])

    def call(self, X):
        return self.model(X)

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

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img)
        features_distorted = self(dist_img)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}

class PerceptNetRegressor(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False, avg_pooling=True, features_diff=False):
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
        self.flatten = layers.GlobalAveragePooling2D() if avg_pooling else layers.Flatten()
        self.regressor = layers.Dense(1)
        self.correlation_loss = PearsonCorrelation()
        self.features_diff = features_diff

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
        if self.features_diff:
            features1 = self.flatten(features1)
            features2 = self.flatten(features2)
            output = tf.math.abs(features1-features2)
        else:
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
            if self.features_diff:
                features_original_f = self.flatten(features_original)
                features_distorted_f = self.flatten(features_distorted)
                features = tf.math.abs(features_original_f-features_distorted_f)
            else:
                features = layers.concatenate([features_original, features_distorted])
                features = self.flatten(features)
            mos_pred = self.regressor(features)
            loss = self.compiled_loss(mos, mos_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ## Evaluating the Pearson Correlation PerceptNet-like
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        correlation = self.correlation_loss(mos, l2)

        ## Evaluating the Pearson Correlation between the MOSes
        correlation_mos = self.correlation_loss(mos, mos_pred)

        return {'loss':loss, 'pearson':correlation, 'pearson_mos':correlation_mos}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self.extract_features(img)
        features_distorted = self.extract_features(dist_img)
        if self.features_diff:
            features_original_f = self.flatten(features_original)
            features_distorted_f = self.flatten(features_distorted)
            features = tf.math.abs(features_original_f-features_distorted_f)
        else:
            features = layers.concatenate([features_original, features_distorted])
            features = self.flatten(features)
        mos_pred = self.regressor(features)
        loss = self.compiled_loss(mos, mos_pred)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        correlation = self.correlation_loss(mos, l2)
        correlation_mos = self.correlation_loss(mos, mos_pred)
        return {'loss':loss, 'pearson':correlation, 'pearson_mos':correlation_mos}

class PerceptNetRegressorFine(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False, avg_pooling=True):
        super(PerceptNetRegressorFine, self).__init__()
        # self.gdn1 = GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer)
        # self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        # self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        # self.gdn2 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        # self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        # self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        # self.gdn3 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        # self.conv3 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same')
        # self.gdn4 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.perceptnet = PerceptNet(kernel_initializer=kernel_initializer,
                                     gdn_kernel_size=gdn_kernel_size,
                                     learnable_undersampling=learnable_undersampling)
        self.flatten = layers.GlobalAveragePooling2D() if avg_pooling else layers.Flatten()
        self.regressor = layers.Dense(1)
        self.correlation_loss = PearsonCorrelation()

    def extract_features(self, X):
        # features = self.gdn1(X)
        # features = self.conv1(features)
        # features = self.undersampling1(features)
        # features = self.gdn2(features)
        # features = self.conv2(features)
        # features = self.undersampling2(features)
        # features = self.gdn3(features)
        # features = self.conv3(features)
        # features = self.gdn4(features)
        features = self.perceptnet(X)
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

# class PerceptNetRandomGabor(BasePercetNet, tf.keras.Model):
#     def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
#         super(tf.keras.Model, self).__init__(**kwargs)
#         super(PerceptNetRandomGabor, self).__init__(feature_extractor=tf.keras.Sequential([
#                                                     GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
#                                                     layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
#                                                     layers.MaxPool2D(2),
#                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                                                     layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
#                                                     layers.MaxPool2D(2),
#                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                                                     RandomGabor(n_gabors=128, size=20),
#                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
#                                                 ]), **kwargs)

class PerceptNetRandomGabor(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False, normalize=True, n_gabors=128):
        super(PerceptNetRandomGabor, self).__init__()
        self.gdn1 = GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer)
        self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn2 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn3 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv3 = RandomGabor(n_gabors=n_gabors, size=20, normalize=normalize)
        self.gdn4 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)

    def call(self, X, training=False):
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output, training=training)
        output = self.gdn4(output)
        return output

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self(img, training=True)
            features_distorted = self(dist_img, training=True)
            l2 = (features_original-features_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img, training=False)
        features_distorted = self(dist_img, training=False)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}

class PerceptNetPseudoRandomGabor(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False, normalize=True, n_gabors=128):
        super(PerceptNetPseudoRandomGabor, self).__init__()
        self.gdn1 = GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer)
        self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn2 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn3 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
        self.conv3 = PseudoRandomGabor(n_gabors=n_gabors, size=20, normalize=normalize)
        self.gdn4 = GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)

    def call(self, X, training=False):
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output, training=training)
        output = self.gdn4(output)
        return output

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self(img, training=True)
            features_distorted = self(dist_img, training=True)
            l2 = (features_original-features_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img, training=False)
        features_distorted = self(dist_img, training=False)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}


class PerceptNetFullRandomGabor(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetFullRandomGabor, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                        RandomGabor(n_gabors=6, size=5),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                        RandomGabor(n_gabors=128, size=20),
                                                        GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
                                                ]), **kwargs)

class PerceptNetExpGaborLast(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExpGaborLast, self).__init__(feature_extractor=tf.keras.Sequential([
                                                     GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                     layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     eflayers.GaborLayer(filters=128, kernel_size=20, strides=1, padding="same"),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
                                                ]), **kwargs)
        
class PerceptNetExpCenterSurround(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, cs_kernel_size=21, relu_cs=False, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExpCenterSurround, self).__init__(feature_extractor=tf.keras.Sequential([
                                                     GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                     layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     eflayers.CenterSurroundLayer(filters=6, kernel_size=cs_kernel_size),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same"),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
                                                ]), **kwargs)

class PerceptNetExpCenterSurroundGabor(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, cs_kernel_size=21, gabor_kernel_size=21, relu_cs=False, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExpCenterSurroundGabor, self).__init__(feature_extractor=tf.keras.Sequential([
                                                     GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                     layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     eflayers.CenterSurroundLayer(filters=6, kernel_size=cs_kernel_size),
                                                     layers.MaxPool2D(2),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
                                                     eflayers.GaborLayer(filters=128, kernel_size=gabor_kernel_size, strides=1, padding="same"),
                                                     GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
                                                ]), **kwargs)

class PerceptNetGaussianGDN(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetGaussianGDN, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=RandomGaussian(filters=6, size=gdn_kernel_size, normalize=True)),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        GDNCustom(layer=RandomGaussian(filters=128, size=gdn_kernel_size, normalize=True)),
                                                ]), **kwargs)


class PerceptNetGaussianGDNGaborLast(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetGaussianGDNGaborLast, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=RandomGaussian(filters=6, size=gdn_kernel_size, normalize=True)),
                                                        PseudoRandomGabor(n_gabors=128, size=20, normalize=True),
                                                        GDNCustom(layer=RandomGaussian(filters=128, size=gdn_kernel_size, normalize=True)),
                                                ]), **kwargs)

class PerceptNetExpGDNGaussian(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExpGDNGaussian, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNCustom(layer=eflayers.GaussianLayer(filters=3, kernel_size=gdn_kernel_size, padding="same")),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=eflayers.GaussianLayer(filters=3, kernel_size=gdn_kernel_size, padding="same")),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNCustom(layer=eflayers.GaussianLayer(filters=6, kernel_size=gdn_kernel_size, padding="same")),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        GDNCustom(layer=eflayers.GaussianLayer(filters=128, kernel_size=gdn_kernel_size, padding="same")),
                                                ]), **kwargs)

class PerceptNetExt256(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExt256, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same'),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                ]), **kwargs)

class PerceptNetExt512(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExt512, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=512, kernel_size=5, strides=1, padding='same'),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                ]), **kwargs)

class PerceptNetExt256_512(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetExt256_512, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                        layers.Conv2D(filters=512, kernel_size=5, strides=1, padding='same'),
                                                        GDNJ(kernel_size=gdn_kernel_size, apply_independently=False, kernel_initializer=kernel_initializer),
                                                ]), **kwargs)

class PerceptNetReLU(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetReLU, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.ReLU(),
                                                ]), **kwargs)

class PerceptNetReLUBN(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetReLUBN, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        layers.BatchNormalization(),
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.BatchNormalization(),
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.BatchNormalization(),
                                                        layers.ReLU(),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.BatchNormalization(),
                                                        layers.ReLU(),
                                                ]), **kwargs)

class PerceptNetBN(BasePercetNet, tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, **kwargs):
        super(tf.keras.Model, self).__init__(**kwargs)
        super(PerceptNetBN, self).__init__(feature_extractor=tf.keras.Sequential([
                                                        layers.BatchNormalization(),
                                                        layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.BatchNormalization(),
                                                        layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
                                                        layers.MaxPool2D(2),
                                                        layers.BatchNormalization(),
                                                        layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
                                                        layers.BatchNormalization(),
                                                ]), **kwargs)

class PerceptNetPatch(tf.keras.Model):
    def __init__(self, patch_size, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False):
        super(PerceptNetPatch, self).__init__()
        self.patch_size = patch_size
        self.extract_patches = Rearrange('batch (h1 h2) (w1 w2) c -> batch (h1 w1) h2 w2 c', h2=patch_size[0], w2=patch_size[1])
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
        ## 1. Extract patches
        output = self.extract_patches(X)
        n_patches = output.shape[1]

        ## 2. Stack the patches in batch dim
        output = rearrange(output, 'batch patch h w c -> (batch patch) h w c')

        ## 3. Pass them through the network
        output = self.gdn1(output)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output)
        output = self.gdn4(output)

        ## 4. Unstack the patches
        output = rearrange(output, '(batch patch) h w c -> batch patch h w c', patch=n_patches)

        return output

    def train_step(self, data):
        """
        data: (X, Y)
            -> X: tuple (Original Image, Distorted Image)
            -> Y: float (MOS score)
        """
        ## 1. Split input
        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            ## 2. Obtain transformed patches
            features_original = self(img)
            features_distorted = self(dist_img)

            ## 3. Obtain centroids of patches from the same image
            centroids_original = reduce(features_original, 'batch patch h w c -> batch h w c', reduction='mean')
            centroids_distorted = reduce(features_distorted, 'batch patch h w c -> batch h w c', reduction='mean')

            ## 4. Measure distance between the centroids
            l2 = (centroids_original-centroids_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)

            ## 5. Correlate the distance with the MOS
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img)
        features_distorted = self(dist_img)
        centroids_original = reduce(features_original, 'batch patch h w c -> batch h w c', reduction='mean')
        centroids_distorted = reduce(features_distorted, 'batch patch h w c -> batch h w c', reduction='mean')
        l2 = (centroids_original-centroids_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}

class PerceptNetGDNGaussianGabor(tf.keras.Model):
    def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False, normalize=True, n_gabors=128):
        super(PerceptNetGDNGaussianGabor, self).__init__()
        self.gdn1 = GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
        self.conv1 = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.undersampling1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn2 = GDNCustom(layer=RandomGaussian(filters=3, size=gdn_kernel_size, normalize=True)),
        self.conv2 = layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same')
        self.undersampling2 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu') if learnable_undersampling else layers.MaxPool2D(2)
        self.gdn3 = GDNCustom(layer=RandomGaussian(filters=6, size=gdn_kernel_size, normalize=True)),
        self.conv3 = PseudoRandomGabor(n_gabors=n_gabors, size=20, normalize=normalize)
        self.gdn4 = GDNCustom(layer=RandomGaussian(filters=128, size=gdn_kernel_size, normalize=True)),

    def call(self, X, training=False):
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output, training=training)
        output = self.gdn4(output)
        return output

    def train_step(self, data):
        """
        X: tuple (Original Image, Distorted Image)
        Y: float (MOS score)
        """

        img, dist_img, mos = data

        with tf.GradientTape() as tape:
            features_original = self(img, training=True)
            features_distorted = self(dist_img, training=True)
            l2 = (features_original-features_distorted)**2
            l2 = tf.reduce_sum(l2, axis=[1,2,3])
            l2 = tf.sqrt(l2)
            loss = self.compiled_loss(mos, l2)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        img, dist_img, mos = data
        features_original = self(img, training=False)
        features_distorted = self(dist_img, training=False)
        l2 = (features_original-features_distorted)**2
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        l2 = tf.sqrt(l2)
        loss = self.compiled_loss(mos, l2)
        self.compiled_metrics.update_state(mos, l2)
        return {m.name: m.result() for m in self.metrics}