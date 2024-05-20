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
parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of training epochs.")
parser.add_argument("-b", "--batch-size", type=int, default=16, help="Number of samples per batch.")
parser.add_argument("--lambda", type=float, default=0., help="Lambda coefficient to weight regularization.")

args = parser.parse_args()
args = vars(args)

if args["kld"]:
    METRIC = "KLD"
elif args["js"]:
    METRIC = "JS"
elif args["mse"]:
    METRIC = "MSE"
else:
    METRIC = None

dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
dst_val = TID2013("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/disk/databases/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/databases/IQA//TID/TID2013/", exclude_imgs=[25])

img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

config = {
    "BATCH_SIZE": args["batch_size"],
    "EPOCHS": args["epochs"],
    "LEARNING_RATE": 3e-4,
    "SEED": 42,
    "METRIC": METRIC,
    "MODEL": "Baseline",
    "CS_KERNEL_SIZE": 5,
    "GDNGAUSSIAN_KERNEL_SIZE": 1,
    "N_GABORS": 128,
    "GABOR_KERNEL_SIZE": 5,
    "GDNSPATIOFREQ_KERNEL_SIZE": 1,
    "LAMBDA": args["lambda"],
}
config = ConfigDict(config)
config

wandb.init(project="PerceptNet_KLD",
           name=args["run_name"],
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
                                #  .map(resize)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)\
                            #  .map(resize)

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
        if args["kld"] or args["js"]:
            mean = GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
            std = GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
            # std = nn.Conv(features=config.N_GABORS, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
            # std = -nn.relu(std)
            return mean, std
        elif args["mse"]:
            return GDN(kernel_size=config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)

@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")
    regularization: metrics.Average.from_output("regularization")

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

def kld(mean_p, std_p, mean_q, std_q, axis=(1,2,3)):
    """Assume diagonal covariance matrix and that the input is the logvariance."""
    logstd_p, logstd_q = std_p, std_q
    std_p, std_q = jnp.exp(std_p), jnp.exp(std_q)
    def safe_div(a, b): return a/b #jnp.where(a == b, 1, a/b)
    logdet_p = jnp.sum(logstd_p, axis=axis)
    logdet_q = jnp.sum(logstd_q, axis=axis)
    
    return (logdet_q - logdet_p) + jnp.sum((1/std_q)*(mean_p - mean_q)**2, axis=axis) + jnp.sum(std_p/std_q, axis=axis)

def js(mean_p, std_p, mean_q, std_q, axis=(1,2,3)):
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
            regularization = (jnp.mean(jnp.exp(img_std)**2) + jnp.mean(jnp.exp(img_dist_std)**2))
        
        elif args["mse"]:
            img_pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
            img_dist_pred, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
            ## Calculate the MSE
            dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)
            regularization = 0

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos) + config.LAMBDA*regularization, (updated_state, regularization)
    
    (loss, (updated_state, regularization)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss, regularization=regularization)
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
            dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params), regularization=None)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

def check_trainable(path):
    return False

trainable_tree = freeze(flax.traverse_util.path_aware_map(lambda path, v: "non_trainable" if check_trainable(path)  else "trainable", state.params))

optimizers = {
    "trainable": optax.adam(learning_rate=config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}

tx = optax.multi_transform(optimizers, trainable_tree)

state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
print(f"Total params: {param_count} | Trainable params: {trainable_param_count}")

wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

metrics_history = {
    "train_loss": [],
    "train_regularization": [],
    "val_loss": [],
    "val_regularization": [],
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