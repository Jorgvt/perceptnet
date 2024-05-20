# %%
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

# %%
import os
import argparse
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

parser = argparse.ArgumentParser(description="Trainig a very simple model on TID08 and testing in TID13",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--id", help="WandB Run ID.")
args = parser.parse_args()
args = vars(args)

# %%
# dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst = KADIK10K("/media/disk/databases/BBDD_video_image/Image_Quality/KADIK10K/")
# dst = PIPAL("/media/disk/databases/BBDD_video_image/Image_Quality/PIPAL/")
dst = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K/")
# dst = KADIK10K("/media/databases/IQA/KADIK10K/")

# %%
img, img_dist, mos = next(iter(dst.dataset))
img.shape, img_dist.shape, mos.shape

# %%
id = args["id"]

# %%
api = wandb.Api()
prev_run = api.run(f"jorgvt/PerceptNet_KLD/{id}")

# %%
config = ConfigDict(prev_run.config["_fields"])
config

# %%
for file in prev_run.files():
    file.download(root=prev_run.dir, replace=True)

# %%
wandb.init(project="PerceptNet_KLD",
           name=prev_run.name,
           job_type="evaluate",
           mode="online",
           )
config = config
config

# %%
def resize(img, img_dist, mos):
    h, w = 384, 512
    img = tf.image.resize(img, (h//8, w//8))
    img_dist = tf.image.resize(img_dist, (h//8, w//8))
    return img, img_dist, mos

# %%
dst_rdy = dst.dataset.batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)\
                    #  .map(resize)

# %% [markdown]
# ## Define the model we're going to use
# 
# > It's going to be a very simple model just for demonstration purposes.

# %%
class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=True)(inputs)
        outputs = nn.Conv(features=3, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=6, kernel_size=(config.CS_KERNEL_SIZE,config.CS_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=config.GDNGAUSSIAN_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=config.N_GABORS, kernel_size=(config.GABOR_KERNEL_SIZE,config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        if config.METRIC == "KLD" or config.METRIC == "JS":
            mean = GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
            std = GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
            # std = nn.Conv(features=config.N_GABORS, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
            # std = -nn.relu(std)
            return mean, std
        elif config.METRIC == "MSE":
            return GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)


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
def kld(mean_p, std_p, mean_q, std_q, axis=(1,2,3)):
    """Assume diagonal covariance matrix and that the input is the logvariance."""
    logstd_p, logstd_q = std_p, std_q
    std_p, std_q = jnp.exp(std_p), jnp.exp(std_q)
    def safe_div(a, b): return a/b #jnp.where(a == b, 1, a/b)
    logdet_p = jnp.sum(logstd_p, axis=axis)
    logdet_q = jnp.sum(logstd_q, axis=axis)
    
    return (logdet_q - logdet_p) + jnp.sum((1/std_q)*(mean_p - mean_q)**2, axis=axis) + jnp.sum(std_p/std_q, axis=axis)

# %%
def js(mean_p, std_p, mean_q, std_q, axis=(1,2,3)):
    return (1/2)*(kld(mean_p, std_p, mean_q, std_q, axis) + kld(mean_q, std_q, mean_p, std_p, axis))

# %%
@jax.jit
def compute_distance(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, img_dist, mos = batch
    
    ## Calculate the KLD
    if config.METRIC == "KLD":
        (img_mean, img_std), updated_state = state.apply_fn({"params": state.params, **state.state}, img, mutable=list(state.state.keys()), train=False)
        (img_dist_mean, img_dist_std), updated_state = state.apply_fn({"params": state.params, **state.state}, img_dist, mutable=list(state.state.keys()), train=False)
        dist = kld(img_mean, img_std, img_dist_mean, img_dist_std)
    
    elif config.METRIC == "JS":
        (img_mean, img_std), updated_state = state.apply_fn({"params": state.params, **state.state}, img, mutable=list(state.state.keys()), train=False)
        (img_dist_mean, img_dist_std), updated_state = state.apply_fn({"params": state.params, **state.state}, img_dist, mutable=list(state.state.keys()), train=False)
        dist = js(img_mean, img_std, img_dist_mean, img_dist_std)
    
    elif config.METRIC == "MSE":
        img_pred, updated_state = state.apply_fn({"params": state.params, **state.state}, img, mutable=list(state.state.keys()), train=True)
        img_dist_pred, updated_state = state.apply_fn({"params": state.params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
        ## Calculate the MSE
        dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)
    
    return dist

# %% [markdown]
# ## Load the pretrained model!

# %%
state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384,512,3))

# %%
def check_trainable(path):
    return False
    # return ("A" in path) or ("alpha_achrom" in path) or ("alpha_chrom_rg" in path) or ("alpha_chrom_yb" in path)

# %%
trainable_tree = freeze(flax.traverse_util.path_aware_map(lambda path, v: "non_trainable" if check_trainable(path)  else "trainable", state.params))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
param_count, trainable_param_count

# %%
wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

# %%
optimizers = {
    "trainable": optax.adam(learning_rate=config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}

# %%
tx = optax.multi_transform(optimizers, trainable_tree)

# %%
state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1,384,512,3))

# %%
# Before actually training the model we're going to set up the checkpointer to be able to save our trained models:
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
# Load weights
state = orbax_checkpointer.restore(os.path.join(prev_run.dir,"model-best"), item=state)

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
print(stats.pearsonr(metrics_history["distance"], metrics_history["mos"]), stats.spearmanr(metrics_history["distance"], metrics_history["mos"]))

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


