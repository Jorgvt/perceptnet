# %%
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm.auto import tqdm

from typing import Any, Callable, Sequence, Union
import numpy as np
import scipy.stats as stats

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training import orbax_utils

import optax
import orbax.checkpoint

from clu import metrics
from ml_collections import ConfigDict

from einops import reduce, rearrange
import wandb

from iqadatasets.datasets import *
from paramperceptnet.models import PerceptNet
from paramperceptnet.training import create_train_state

# %%
dst = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008/")

# %%
img, img_dist, mos = next(iter(dst.dataset))
img.shape, img_dist.shape, mos.shape

# %%
ids = ["i8kkltwu", # TrainAll_GoodInit
       "gx9gpizs", # Freeze_GDNGamma
       "c9u2vqjz", # Freeze_J&H
       "2aae1qvd", # Freeze_GDNColor
       "f8uv6afu", # Freeze_CS
       "3r2slksi", # Freeze_GDNGaussian
       "k24dfyo8", # Freeze_GDNFinalOnle (Freeze_Gabor)
       "csrhdpbd", # OnlyB
       ]

for id in tqdm(ids):
    print(id)
    # %%
    api = wandb.Api()
    prev_run = api.run(f"jorgvt/PerceptNet_v15/{id}")
    print(prev_run.name)

    # %%
    try:
        config = ConfigDict(prev_run.config["_fields"])
    except:
        config = ConfigDict(prev_run.config)

    print(config)

    # %%
    for file in prev_run.files():
        file.download(root=prev_run.dir, replace=True)

    # %%
    wandb.init(project="PerceptNet_JaX_Eval",
               name=prev_run.name,
               job_type="evaluate",
               mode="online",
               )

    # %%
    dst_rdy = dst.dataset.batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

    # %% [markdown]
    # ## Define the model we're going to use
    # 
    # > It's going to be a very simple model just for demonstration purposes.

    # %%
    # %%
    @jax.jit
    def compute_distance(*, state, batch):
        """Obtaining the metrics for a given batch."""
        img, img_dist, mos = batch

        ## Forward pass through the model
        img_pred = state.apply_fn({"params": state.params, **state.state}, img, train=False)
        img_dist_pred = state.apply_fn({"params": state.params, **state.state}, img_dist, train=False)

        ## Calculate the distance
        dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)

        ## Calculate pearson correlation
        return dist

    # %% [markdown]
    # ## Load the pretrained model!

    # %%
    state = create_train_state(PerceptNet(config), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384,512,3))

    # %%
    # Before actually training the model we're going to set up the checkpointer to be able to save our trained models:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)

    # %%
    a = orbax_checkpointer.restore(os.path.join(prev_run.dir,"model-best"))
    try:
        state = state.replace(params=a["params"],
                              state=a["state"])
    except:
        state = state.replace(params=a["params"])

    print("Parameters loaded!")

    # %% [markdown]
    # ## Evaluate!

    # %%
    metrics_history = {
        "distance": [],
        "mos": [],
    }

    # %%
    
    for batch in tqdm(dst_rdy.as_numpy_iterator()):
        img, img_dist, mos = batch
        distance = compute_distance(state=state, batch=batch)
        metrics_history["distance"].extend(distance)
        metrics_history["mos"].extend(mos)
        # break

    # %%
    assert len(metrics_history["distance"]) == len(dst.data)

    # %%
    stats.pearsonr(metrics_history["distance"], metrics_history["mos"]), stats.spearmanr(metrics_history["distance"], metrics_history["mos"])

    # %%
    results = dst.data.copy()
    results["Distance"] = metrics_history["distance"]
    results.head()

    # %%
    wandb.log({"TID2008": wandb.Table(dataframe=results),
               "TID2008_pearson": stats.pearsonr(metrics_history["distance"], metrics_history["mos"])[0],
               "TID2008_spearman": stats.spearmanr(metrics_history["distance"], metrics_history["mos"])[0],
               })

    # %%
    wandb.finish()


