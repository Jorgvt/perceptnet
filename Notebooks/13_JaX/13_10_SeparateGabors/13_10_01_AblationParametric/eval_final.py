import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

from typing import Any, Callable, Sequence, Union
import numpy as np

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
from ml_collections import ConfigDict, config_flags

from einops import reduce, rearrange, repeat
import wandb
from iqadatasets.datasets import *
from fxlayers.layers import *
from fxlayers.layers import GaussianLayerGamma, FreqGaussianGamma, OrientGaussianGamma, GaborLayerGamma_, GaborLayerGammaRepeat
from fxlayers.initializers import *
from JaxPlayground.utils.constraints import *
from JaxPlayground.utils.wandb import *

from paramperceptnet.layers import *
from paramperceptnet.training import *
from tqdm.auto import tqdm
import pandas as pd
import scipy.stats as stats

class GDNSpatioChromaFreqOrient(nn.Module):
    """Generalized Divisive Normalization."""
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    # inputs_star: float = 1.
    # outputs_star: Union[None, float] = None
    fs: int = 1
    apply_independently: bool = False
    bias_init: Callable = nn.initializers.ones_init()
    alpha: float = 2.
    epsilon: float = 1/2 # Exponential of the denominator
    eps: float = 1e-6 # Numerical stability in the denominator

    @nn.compact
    def __call__(self,
                 inputs,
                 fmean,
                 theta_mean,
                 train=False,
                 ):
        b, h, w, c = inputs.shape
        bias = self.param("bias",
                          #equal_to(inputs_star/10),
                          self.bias_init,
                          (c,))
        # is_initialized = self.has_variable("batch_stats", "inputs_star")
        # inputs_star = self.variable("batch_stats", "inputs_star", lambda x: jnp.ones(x)*self.inputs_star, (len(self.inputs_star),))
        # inputs_star_ = jnp.ones_like(inputs)*inputs_star.value
        GL = GaussianLayerGamma(features=c, kernel_size=self.kernel_size, strides=self.strides, padding="VALID", fs=self.fs, xmean=self.kernel_size/self.fs/2, ymean=self.kernel_size/self.fs/2, normalize_prob=config.NORMALIZE_PROB, normalize_energy=config.NORMALIZE_ENERGY, use_bias=False, feature_group_count=c)
        FOG = ChromaFreqOrientGaussianGamma()
        outputs = GL(pad_same_from_kernel_size(inputs, kernel_size=self.kernel_size, mode=self.padding)**self.alpha, train=train)#/(self.kernel_size**2)
        outputs = FOG(outputs, fmean=fmean, theta_mean=theta_mean)

        ## Coef
        # coef = GL(inputs_star_**self.alpha, train=train)#/(self.kernel_size**2)
        # coef = FG(coef, fmean=fmean)
        # coef = rearrange(coef, "b h w (phase theta f) -> b h w (phase f theta)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = OG(coef, theta_mean=theta_mean) + bias
        # coef = rearrange(coef, "b h w (phase f theta) -> b h w (phase theta f)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = jnp.clip(coef+bias, a_min=1e-5)**self.epsilon
        # # coef = inputs_star.value * coef
        # if self.outputs_star is not None: coef = coef/inputs_star.value*self.outputs_star

        # if is_initialized and train:
        #     inputs_star.value = (inputs_star.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(0,1,2)))/2
        # return coef * inputs / (jnp.clip(denom+bias, a_min=1e-5)**self.epsilon + self.eps)
        return inputs / (jnp.clip(outputs+bias, a_min=1e-5)**self.epsilon + self.eps)

class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""
    config: Any

    @nn.compact
    def __call__(self,
                 inputs, # Assuming fs = 128 (cpd)
                 **kwargs,
                 ):
        ## (Independent) Color equilibration (Gamma correction)
        ## Might need to be the same for each number
        ## bias = 0.1 / kernel = 0.5
        if self.config.USE_GAMMA: outputs = GDNGamma()(inputs)
        else: outputs = GDN(kernel_size=(1,1), apply_independently=True)(inputs)
        
        ## Color (ATD) Transformation
        outputs = nn.Conv(features=3, kernel_size=(1,1), use_bias=False, name="Color")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        
        ## GDN Star A - T - D [Separated]
        outputs = GDN(kernel_size=(1,1), apply_independently=True)(outputs)

        ## Center Surround (DoG)
        ## Initialized so that 3 are positives and 3 are negatives and no interaction between channels is present
        outputs = pad_same_from_kernel_size(outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric")
        if self.config.PARAM_CS:
            outputs = CenterSurroundLogSigmaK(features=3, kernel_size=self.config.CS_KERNEL_SIZE, fs=21, use_bias=False, padding="VALID")(outputs, **kwargs)
        else:
            outputs = nn.Conv(features=3, kernel_size=(self.config.CS_KERNEL_SIZE,self.config.CS_KERNEL_SIZE), use_bias=False, padding="VALID")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))

        ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
        ### fs = 32 / kernel_size = (11,11) -> 0.32 > 0.02 --> OK!
        ## TO-DO: - Spatial Gaussian Kernel (0.02 deg) -> fs = 64/2 & 0.02*64/2 = sigma (px) = 0.69
        if self.config.PARAM_DN_CS:
            outputs = GDNGaussian(kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, apply_independently=True, fs=32, padding="symmetric", normalize_prob=self.config.NORMALIZE_PROB, normalize_energy=self.config.NORMALIZE_ENERGY)(outputs, **kwargs)

        else:
            # outputs = pad_same_from_kernel_size(outputs, kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, mode="symmetric")
            outputs = GDN(kernel_size=(self.config.GDNGAUSSIAN_KERNEL_SIZE,self.config.GDNGAUSSIAN_KERNEL_SIZE), apply_independently=True, padding="SAME")(outputs)
        ## GaborLayer per channel with GDN mixing only same-origin-channel information
        ### [Gaussian] sigma = 0.2 (deg) fs = 32 / kernel_size = (21,21) -> 21/32 = 0.66 --> OK!
        outputs = pad_same_from_kernel_size(outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric")
        # outputs, fmean, theta_mean = GaborLayerGamma_(n_scales=4+2+2, n_orientations=8*3, kernel_size=self.config.GABOR_KERNEL_SIZE, fs=32, xmean=self.config.GABOR_KERNEL_SIZE/32/2, ymean=self.config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=self.config.NORMALIZE_PROB, normalize_energy=self.config.NORMALIZE_ENERGY, zero_mean=self.config.ZERO_MEAN, use_bias=self.config.USE_BIAS, train_A=self.config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        if self.config.PARAM_GABOR:
            outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(n_scales=[4,2,2], n_orientations=[8,8,8], kernel_size=self.config.GABOR_KERNEL_SIZE, fs=32, xmean=self.config.GABOR_KERNEL_SIZE/32/2, ymean=self.config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=self.config.NORMALIZE_PROB, normalize_energy=self.config.NORMALIZE_ENERGY, zero_mean=self.config.ZERO_MEAN, use_bias=self.config.USE_BIAS, train_A=self.config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        else:
            outputs = pad_same_from_kernel_size(outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric")
            outputs = nn.Conv(features=128, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), padding="VALID", use_bias=False)(outputs)
        ## Final GDN mixing Gabor information (?)
        if self.config.PARAM_DN_FINAL:
            outputs = GDNSpatioChromaFreqOrient(kernel_size=21, strides=1, padding="symmetric", fs=32, apply_independently=False)(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)
        else:
            # outputs = pad_same_from_kernel_size(outputs, kernel_size=21, mode="symmetric")
            outputs = GDN(kernel_size=(21,21), apply_independently=False, padding="SAME")(outputs)

        if self.config.FINAL_B: B = self.param("B", nn.initializers.ones_init(), (outputs.shape[-1],))
        else: B = 1.
        return B*outputs

# %%
dst = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008/")


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

df = pd.read_csv("https://gist.githubusercontent.com/Jorgvt/c58e89013246a1a69ac9d01f34b2dfcc/raw/44a9c3f8598a3ae6b7506b445ea727ce5d1e6822/gistfile1.txt")
# df = pd.read_csv("https://gist.githubusercontent.com/Jorgvt/c579fc2d684662fa5c692540d88c4fac/raw/c73d7707e519340713df76fcedf048284782566d/gistfile1.txt")
ids = df.id.to_list()
# ids = ["ymt8elgp"]
total = len(ids)

for i, run_id in enumerate(ids, 1):

    api = wandb.Api()
    prev_run = api.run(f"jorgvt/PerceptNet_v15/{run_id}")
    print(prev_run.name)

    # %%
    try:
        config = ConfigDict(prev_run.config["_fields"])
    except:
        config = ConfigDict(prev_run.config)

    print(config)


    dst_rdy = dst.dataset.batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

    # %%
    for file in prev_run.files():
        file.download(root=prev_run.dir, replace=True)

    # %%
    wandb.init(project="PerceptNet_JaX_Eval",
               name=prev_run.name,
               job_type="evaluate",
               mode="disabled",
               )
    wandb.run.summary["train_id"] = run_id
    
    state = create_train_state(PerceptNet(config), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384,512,3))
    def check_trainable(path):
        if not config.A_GDNSPATIOFREQORIENT:
            if ("GDNSpatioChromaFreqOrient_0" in path) and ("A" in path):
                return True
        if "Color" in path:
            if not config.TRAIN_JH:
                return True
        if "CenterSurroundLogSigmaK_0" in path:
            if not config.TRAIN_CS:
                return True
        if "Gabor" in "".join(path):
            if not config.TRAIN_GABOR:
                return True
        if "GDNSpatioChromaFreqOrient_0" not in path and config.TRAIN_ONLY_LAST_GDN:
            return True
        return False

    # %%
    trainable_tree = freeze(flax.traverse_util.path_aware_map(lambda path, v: "non_trainable" if check_trainable(path)  else "trainable", state.params))

    # %%
    tx = optax.adam(learning_rate=config.LEARNING_RATE)
    optimizers = {
        "trainable": tx,
        "non_trainable": optax.set_to_zero(),
    }

    # %%
    tx = optax.multi_transform(optimizers, trainable_tree)

    # %%
    state = create_train_state(PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1,384,512,3))

    # %%
    # Before actually training the model we're going to set up the checkpointer to be able to save our trained models:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)

    # %%
    state = orbax_checkpointer.restore(os.path.join(prev_run.dir,"model-best"), item=state)
    # a = orbax_checkpointer.restore(os.path.join(prev_run.dir,"model-best"))
    # params_ = freeze(a["params"])
    # state_ = freeze(a["state"])
    # continue

    # try:
    #     state = state.replace(params=params_,
    #                           state=state_)

    # except:
    #     state = state.replace(params=params_)


    print("Parameters loaded!")
    if state.state is None:

        @jax.jit
        def compute_distance(*, state, batch):
            """Obtaining the metrics for a given batch."""
            img, img_dist, mos = batch

            ## Forward pass through the model
            img_pred = state.apply_fn({"params": state.params}, img, train=False)
            img_dist_pred = state.apply_fn({"params": state.params}, img_dist, train=False)

            ## Calculate the distance
            dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)

            ## Calculate pearson correlation
            return dist

    metrics_history = {
        "distance": [],
        "mos": [],
    }

    # %%
    
    for batch in tqdm(dst_rdy.as_numpy_iterator(), leave=False):
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
    print(f"{i}/{total}")
