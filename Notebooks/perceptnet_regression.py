from collections import namedtuple
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from glob import glob
import cv2
import wandb
from wandb.keras import WandbCallback

from GDN_Jorge import GDN as GDNJ
from networks import PerceptNetRegressor
from pearson_loss import PearsonCorrelation
from callbacks import GDNWeightWatcherWandb

def filter_data_2008(img_ids='all', 
                dist_ids='all',
                dist_ints='all',
                exclude_img_ids=None,
                exclude_dist_ids=None,
                exclude_dist_ints=None):
    """
    Filters the data to only utilize a subset based on img_id.

    Parameters
    ----------
    img_ids: list[string]
        List of image IDs to use passed as strings.
    dist_ids: list[string]
        List of image IDs to use passed as strings.
    dist_int: list[string]
        List of image IDs to use passed as strings.
        As of now, the intensities go from 1 to 5.
    exclude_img_ids: list[string]
        List of image IDs to exclude passed as strings.
    exclude_dist_ids: list[string]
        List of image IDs to exclude passed as strings.
    exclude_dist_int: list[string]
        List of image IDs to exclude passed as strings.
        As of now, the intensities go from 1 to 5.

    Returns
    -------
    data: list[ImagePair]
        List of ImagePair objects containing the paths to the image pairs and 
        their corresponding metric.
    """
    ## It's not good practice to default a parameter as an empty list.
    ## The good practice is to default it as a None and then create the empty list.
    exclude_img_ids = [] if exclude_img_ids == None else exclude_img_ids
    exclude_dist_ids = [] if exclude_dist_ids == None else exclude_dist_ids
    exclude_dist_ints = [] if exclude_dist_ints == None else exclude_dist_ints
    data = []
    for img_path in glob(os.path.join(path_2008, 'reference_images', '*.BMP')):
        if img_ids != 'all': # We only want to skip images if any ids were specified
            if img_path.lower().split("/")[-1].split(".")[0][1:] not in img_ids:
                continue # Skips this loop iteration if the ids is not being selected
        elif len(exclude_img_ids)>0:
            if img_path.lower().split("/")[-1].split(".")[0][1:] in exclude_img_ids:
                continue
        for dist_img_path in glob(os.path.join(path_2008, 'distorted_images', f'{img_path.split("/")[-1].split(".")[0]}*')):
            dist_id, dist_int = dist_img_path.lower().split("/")[-1].split(".")[0][1:].split("_")[1:]
            
            if dist_ids!='all':
                if dist_id not in dist_ids:
                    continue
            elif len(exclude_dist_ids)>0:
                if dist_id in exclude_dist_ids:
                    continue
            if dist_ints!='all':
                if dist_int not in dist_ints:
                    continue
            elif len(exclude_dist_ints)>0:
                if dist_int in exclude_dist_ints:
                    continue
            
            data.append(ImagePair(img_path, dist_img_path, name_metric_2008[dist_img_path.split("/")[-1].split(".")[0]]))
    return data

def filter_data_2013(img_ids='all', 
                dist_ids='all',
                dist_ints='all',
                exclude_img_ids=None,
                exclude_dist_ids=None,
                exclude_dist_ints=None):
    """
    Filters the data to only utilize a subset based on img_id.

    Parameters
    ----------
    img_ids: list[string]
        List of image IDs to use passed as strings.
    dist_ids: list[string]
        List of image IDs to use passed as strings.
    dist_int: list[string]
        List of image IDs to use passed as strings.
        As of now, the intensities go from 1 to 5.
    exclude_img_ids: list[string]
        List of image IDs to exclude passed as strings.
    exclude_dist_ids: list[string]
        List of image IDs to exclude passed as strings.
    exclude_dist_int: list[string]
        List of image IDs to exclude passed as strings.
        As of now, the intensities go from 1 to 5.

    Returns
    -------
    data: list[ImagePair]
        List of ImagePair objects containing the paths to the image pairs and 
        their corresponding metric.
    """
    ## It's not good practice to default a parameter as an empty list.
    ## The good practice is to default it as a None and then create the empty list.
    exclude_img_ids = [] if exclude_img_ids == None else exclude_img_ids
    exclude_dist_ids = [] if exclude_dist_ids == None else exclude_dist_ids
    exclude_dist_ints = [] if exclude_dist_ints == None else exclude_dist_ints
    data = []
    for img_path in glob(os.path.join(path_2013, 'reference_images', '*.BMP')):
        if img_ids != 'all': # We only want to skip images if any ids were specified
            if img_path.lower().split("/")[-1].split(".")[0][1:] not in img_ids:
                continue # Skips this loop iteration if the ids is not being selected
        elif len(exclude_img_ids)>0:
            if img_path.lower().split("/")[-1].split(".")[0][1:] in exclude_img_ids:
                continue
        for dist_img_path in glob(os.path.join(path_2013, 'distorted_images', f'{img_path.lower().split("/")[-1].split(".")[0]}*')):
            dist_id, dist_int = dist_img_path.lower().split("/")[-1].split(".")[0][1:].split("_")[1:]
            
            if dist_ids!='all':
                if dist_id not in dist_ids:
                    continue
            elif len(exclude_dist_ids)>0:
                if dist_id in exclude_dist_ids:
                    continue
            if dist_ints!='all':
                if dist_int not in dist_ints:
                    continue
            elif len(exclude_dist_ints)>0:
                if dist_int in exclude_dist_ints:
                    continue
            
            data.append(ImagePair(img_path, dist_img_path, name_metric_2013[dist_img_path.split("/")[-1].split(".")[0]]))
    return data

# class PerceptNet(tf.keras.Model):
#     def __init__(self, kernel_initializer='identity', gdn_kernel_size=1, learnable_undersampling=False):
#         super(PerceptNet, self).__init__()
#         if learnable_undersampling:
#             self.model = tf.keras.Sequential([
#                 GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
#                 layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu'),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
#                 layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=1, activation='relu'),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
#             ])
#         else:
#             self.model = tf.keras.Sequential([
#                 GDNJ(kernel_size=gdn_kernel_size, apply_independently=True, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
#                 layers.MaxPool2D(2),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same'),
#                 layers.MaxPool2D(2),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer),
#                 layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
#                 GDNJ(kernel_size=gdn_kernel_size, kernel_initializer=kernel_initializer)
#             ])
    
#     def call(self, X):
#         return self.model(X)

#     def train_step(self, data):
#         """
#         X: tuple (Original Image, Distorted Image)
#         Y: float (MOS score)
#         """

#         img, dist_img, mos = data

#         with tf.GradientTape() as tape:
#             features_original = self(img)
#             features_distorted = self(dist_img)
#             l2 = (features_original-features_distorted)**2
#             l2 = tf.reduce_sum(l2, axis=[1,2,3])
#             l2 = tf.sqrt(l2)
#             loss = self.compiled_loss(mos, l2)
        
#         gradients = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

#         return {'pearson':loss}
    
#     def test_step(self, data):
#         img, dist_img, mos = data
#         features_original = self(img)
#         features_distorted = self(dist_img)
#         l2 = (features_original-features_distorted)**2
#         l2 = tf.reduce_sum(l2, axis=[1,2,3])
#         l2 = tf.sqrt(l2)
#         loss = self.compiled_loss(mos, l2)
#         return {'pearson':loss}


if __name__ == '__main__':

    path_2008 = '/media/disk/databases/BBDD_video_image/Image_Quality/TID/TID2008'
    path_2013 = '/media/disk/databases/BBDD_video_image/Image_Quality/TID/TID2013'

    name_metric_2008 = {}
    with open(os.path.join(path_2008, 'mos_with_names.txt')) as f:
        for line in f.readlines():
            # remove last character to avoid \n
            metric, file_name = line[:-1].split(" ")
            name_metric_2008[file_name.upper().split(".")[0]] = float(metric)

    name_metric_2013 = {}
    with open(os.path.join(path_2013, 'mos_with_names.txt')) as f:
        for line in f.readlines():
            # remove last character to avoid \n
            metric, file_name = line[:-1].split(" ")
            name_metric_2013[file_name.lower().split(".")[0]] = float(metric)

    ImagePair = namedtuple('ImagePair', 'img_path dist_img_path metric')

    config = {
        'seed':42,
        'train_dataset':'TID2008',
        'test_dataset':'TID2013',
        'epochs':150,
        'learning_rate':3e-4,
        'batch_size':32,
        'kernel_initializer':'zeros',
        'gdn_kernel_size':1,
        'learnable_undersampling':True,
        # 'test_images':['20', '21', '22', '23', '24'],
        # 'test_dists':['05', '10', '15', '20', '24']
    }

    wandb.init(project='PerceptNetRegressor',
               notes="Excluding non-natural image.",
               tags=["mse", "excluded non-natural", 'Train_TID2008', 'Test_TID2013'],
               name = 'LearnableUndersampling',
               config=config,
               mode='online') # 'disabled'/'online'
    config = wandb.config
    
    wandb.run.name = f'{wandb.run.name}_GDN_Kernel_Size={config.gdn_kernel_size}'
    wandb.run.save()

    # tf.keras.utils.set_random_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    train_data = filter_data_2008(img_ids='all',
                             dist_ids='all',
                             dist_ints='all',
                             exclude_img_ids=['25'],
                       )
    test_data = filter_data_2013(img_ids='all',
                            dist_ids='all',
                            dist_ints='all',
                            exclude_img_ids=['25'],
                       )

    def train_gen():
        for sample in train_data:
            img = cv2.imread(sample.img_path)/255.0
            dist_img = cv2.imread(sample.dist_img_path)/255.0
            metric = sample.metric
            yield img, dist_img, metric

    def test_gen():
        for sample in test_data:
            img = cv2.imread(sample.img_path)/255.0
            dist_img = cv2.imread(sample.dist_img_path)/255.0
            metric = sample.metric
            yield img, dist_img, metric

    train_dataset = tf.data.Dataset.from_generator(train_gen,
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(), dtype=tf.float32)
                                                   ))
    test_dataset = tf.data.Dataset.from_generator(test_gen,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(), dtype=tf.float32)
                                                  ))
    
    
    model = PerceptNetRegressor(kernel_initializer=config.kernel_initializer, 
                                gdn_kernel_size=config.gdn_kernel_size, 
                                learnable_undersampling=config.learnable_undersampling)
    model.compile(optimizer=Adam(learning_rate=config.learning_rate),
                  loss='mse')

    history = model.fit(train_dataset.shuffle(buffer_size=50,
                                              reshuffle_each_iteration=True,
                                              seed=42) \
                                     .batch(config.batch_size), 
                        epochs=config.epochs,
                        verbose=0,
                        validation_data=test_dataset.batch(config.batch_size),
                        callbacks=[WandbCallback(monitor='val_pearson',
                                                 mode='min',
                                                 save_model=True,
                                                 save_weights_only=True,
                                                 log_weights=True),
                                   GDNWeightWatcherWandb()])
    wandb.run.summary['n_params'] = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    wandb.finish()
