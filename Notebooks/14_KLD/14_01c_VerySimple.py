import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import os
from typing import Any, Callable, Sequence, Union
import argparse

import numpy as np
from einops import rearrange

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

from einops import reduce
import wandb
from iqadatasets.datasets import *
from fxlayers.layers import *
from JaxPlayground.utils.constraints import *
from JaxPlayground.utils.wandb import *

parser = argparse.ArgumentParser(description="Trainig a very simple model on TID08 and testing in TID13",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mse", action="store_true", help="Use MSE")
parser.add_argument("--kld", action="store_true", help="Use KLD")
parser.add_argument("--js", action="store_true", help="Use JS")
parser.add_argument("--testing", action="store_true", help="Perform only one batch of training and one of validation.")
parser.add_argument("--wandb", default="disabled", help="WandB mode.")
parser.add_argument("--run_name", default=None, help="Name for the WandB run.")

args = parser.parse_args()
args = vars(args)

# dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
dst_train = TID2008("/media/databases/IQA//TID/TID2008/", exclude_imgs=[25])
dst_val = TID2013("/media/databases/IQA//TID/TID2013/", exclude_imgs=[25])

img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

config = {
    "BATCH_SIZE": 32,
    "EPOCHS": 30,
    "LEARNING_RATE": 3e-4,
    "SEED": 42,
}
config = ConfigDict(config)
config

wandb.init(project="PerceptNet_KLD",
           name=args["name"],
           job_type="training",
           config=config,
           mode=args["wandb"],
           )
config = config
config

def resize(img, img_dist, mos):
    h, w = 384, 512
    img = tf.image.resize(img, (h//8, w//8))
    img_dist = tf.image.resize(img_dist, (h//8, w//8))
    return img, img_dist, mos


dst_train_rdy = dst_train.dataset.shuffle(buffer_size=100,
                                      reshuffle_each_iteration=True,
                                      seed=config.SEED)\
                                 .batch(config.BATCH_SIZE, drop_remainder=True)\
                                 .map(resize)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)\
                             .map(resize)

class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = rearrange(inputs, "b h w c -> b (h w c)")
        if args["kld"] or args["js"]:
            mean = nn.Dense(features=2)(outputs)
            std = nn.Dense(features=2)(outputs)
            return mean, std
        elif args["mse"]:
            return nn.Dense(features=2)(outputs)

@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")

class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict

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

def pearson_correlation(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    num = vec1-vec1_mean
    num *= vec2-vec2_mean
    num = num.sum()
    denom = jnp.sqrt(jnp.sum((vec1-vec1_mean)**2))
    denom *= jnp.sqrt(jnp.sum((vec2-vec2_mean)**2))
    return num/denom

def kld(mean_p, std_p, mean_q, std_q, axis=(1)):
    """Assume diagonal covariance matrix and that the input is the logvariance."""
    std_p, std_q = jnp.exp(std_p), jnp.exp(std_q)
    def safe_div(a, b): return a/b #jnp.where(a == b, 1, a/b)
    det_p = jnp.prod(std_p, axis=axis) + 1e-5
    det_q = jnp.prod(std_q, axis=axis) + 1e-5
    
    return jnp.log(safe_div(det_p, det_q)) + jnp.sum((1/std_q)*(mean_p - mean_q)**2, axis=axis) + jnp.sum(std_p/std_q, axis=axis)

def js(mean_p, std_p, mean_q, std_q, axis=(1)):
    return (1/2)*(kld(mean_p, std_p, mean_q, std_q, axis) + kld(mean_q, std_q, mean_p, std_p, axis))

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        if args["kld"] or args["js"]:
            (img_mean, img_std), updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
            (img_dist_mean, img_dist_std), updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
            ## Calculate the KLD
            if args["kld"]: dist = kld(img_mean, img_std, img_dist_mean, img_dist_std)
            if args["js"]: dist = js(img_mean, img_std, img_dist_mean, img_dist_std)
        
        elif args["mse"]:
            img_pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
            img_dist_pred, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
            ## Calculate the MSE
            dist = ((img_pred - img_dist_pred)**2).sum(axis=(1))**(1/2)

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos), updated_state
    
    (loss, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    state = state.replace(state=updated_state)
    return state

@jax.jit
def compute_metrics(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        if args["kld"] or args["js"]:
            (img_mean, img_std), updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
            (img_dist_mean, img_dist_std), updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
            ## Calculate the KLD
            if args["kld"]: dist = kld(img_mean, img_std, img_dist_mean, img_dist_std)
            if args["js"]: dist = js(img_mean, img_std, img_dist_mean, img_dist_std)
        
        elif args["mse"]:
            img_pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
            img_dist_pred, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
            ## Calculate the MSE
            dist = ((img_pred - img_dist_pred)**2).sum(axis=(1))**(1/2)

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params))
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384//8,512//8,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

def check_trainable(path):
    return False

trainable_tree = freeze(flax.traverse_util.path_aware_map(lambda path, v: "non_trainable" if check_trainable(path)  else "trainable", state.params))

optimizers = {
    "trainable": optax.adam(learning_rate=config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}

tx = optax.multi_transform(optimizers, trainable_tree)

state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1,384//8,512//8,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
param_count, trainable_param_count

wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state = train_step(state, batch)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        if args["testing"]: break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)
    
    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        if args["testing"]: break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())
    
    ## Checkpointing
    if not args["testing"]:
        if metrics_history["val_loss"][-1] <= max(metrics_history["val_loss"]):
            orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-best"), state, save_args=save_args, force=True) # force=True means allow overwritting.

    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()}, commit=False)
    wandb.log({"epoch": epoch+1, **{name:values[-1] for name, values in metrics_history.items()}})
    print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}')
    if args["testing"]: break

if not args["testing"]: orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args)

wandb.finish()