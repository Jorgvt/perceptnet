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
from fxlayers.layers import *
from fxlayers.layers import GaborLayerLogSigma_, GaussianLayerGamma, FreqGaussianGamma, OrientGaussianGamma
from fxlayers.initializers import *
from JaxPlayground.utils.constraints import *
from JaxPlayground.utils.wandb import *

# %%
dst = KADIK10K("/media/disk/vista/BBDD_video_image/Image_Quality/KADIK10K/")

# %%
img, img_dist, mos = next(iter(dst.dataset))
img.shape, img_dist.shape, mos.shape

# %%
ids = ["afr86ups"]

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


    class PerceptNet(nn.Module):
        """IQA model inspired by the visual system."""

        @nn.compact
        def __call__(self,
                     inputs, # Assuming fs = 128 (cpd)
                     **kwargs,
                     ):
            ## (Independent) Color equilibration (Gamma correction)
            ## Might need to be the same for each number
            ## bias = 0.1 / kernel = 0.5
            if config.USE_GAMMA: outputs = GDNGamma()(inputs)
            else: outputs = GDN(kernel_size=(1,1), apply_independently=True)(inputs)

            ## Color (ATD) Transformation
            outputs = nn.Conv(features=3, kernel_size=(1,1), use_bias=False, name="Color")(outputs)
            outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))

            ## GDN Star A - T - D [Separated]
            outputs = GDN(kernel_size=(1,1), apply_independently=True)(outputs)

            ## Center Surround (DoG)
            ## Initialized so that 3 are positives and 3 are negatives and no interaction between channels is present
            outputs = pad_same_from_kernel_size(outputs, kernel_size=config.CS_KERNEL_SIZE, mode="symmetric")
            outputs = nn.Conv(features=3, kernel_size=(config.CS_KERNEL_SIZE,config.CS_KERNEL_SIZE), use_bias=False, padding="VALID")(outputs)
            outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))

            ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
            ### fs = 32 / kernel_size = (11,11) -> 0.32 > 0.02 --> OK!
            ## TO-DO: - Spatial Gaussian Kernel (0.02 deg) -> fs = 64/2 & 0.02*64/2 = sigma (px) = 0.69
            outputs = GDN(kernel_size=(config.GDNGAUSSIAN_KERNEL_SIZE,config.GDNGAUSSIAN_KERNEL_SIZE), apply_independently=True, padding="SAME")(outputs)

            ## GaborLayer per channel with GDN mixing only same-origin-channel information
            ### [Gaussian] sigma = 0.2 (deg) fs = 32 / kernel_size = (21,21) -> 21/32 = 0.66 --> OK!
            outputs = pad_same_from_kernel_size(outputs, kernel_size=config.GABOR_KERNEL_SIZE, mode="symmetric")
            # outputs, fmean, theta_mean = GaborLayerGamma_(n_scales=4+2+2, n_orientations=8*3, kernel_size=config.GABOR_KERNEL_SIZE, fs=32, xmean=config.GABOR_KERNEL_SIZE/32/2, ymean=config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=config.NORMALIZE_PROB, normalize_energy=config.NORMALIZE_ENERGY, zero_mean=config.ZERO_MEAN, use_bias=config.USE_BIAS, train_A=config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
            outputs = nn.Conv(features=128, kernel_size=(config.GABOR_KERNEL_SIZE,config.GABOR_KERNEL_SIZE), padding="VALID", use_bias=False)(outputs)
            ## Final GDN mixing Gabor information (?)
            outputs = GDN(kernel_size=(21,21), apply_independently=False, padding="SAME")(outputs)
    
            return outputs

    # %% [markdown]
    # ## Define the metrics with `clu`

    # %%
    @struct.dataclass
    class Metrics(metrics.Collection):
        """Collection of metrics to be tracked during training."""
        loss: metrics.Average.from_output("loss")

    # %% [markdown]
    # By default, `TrainState` doesn't include metrics, but it's very easy to subclass it so that it does:

    # %%
    class TrainState(train_state.TrainState):
        metrics: Metrics
        state: FrozenDict

    # %% [markdown]
    # We'll define a function that initializes the `TrainState` from a module, a rng key and some optimizer:

    # %%
    def create_train_state(module, key, tx, input_shape):
        """Creates the initial `TrainState`."""
        variables = module.init(key, jnp.ones(input_shape))
        state, params = variables.pop('params')
        return TrainState.create(
            apply_fn=module.apply,
            params=params,
            state=state,
            tx=tx,
            metrics=Metrics.empty()
        )

    # %% [markdown]
    # ## Define evaluation step

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
    state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384,512,3))

    # %%
    # Before actually training the model we're going to set up the checkpointer to be able to save our trained models:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)

    # %%
    a = orbax_checkpointer.restore(os.path.join(prev_run.dir,"model-best"))
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
    wandb.log({"KADID10K": wandb.Table(dataframe=results),
               "KADID10K_pearson": stats.pearsonr(metrics_history["distance"], metrics_history["mos"])[0],
               "KADID10K_spearman": stats.spearmanr(metrics_history["distance"], metrics_history["mos"])[0],
               })

    # %%
    wandb.finish()


