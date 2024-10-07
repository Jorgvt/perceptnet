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



# parser = argparse.ArgumentParser(description="Training with different configurations",
                                #  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-c", "--config", help="Path to the configuration file.")
# parser.add_argument("-n", "--name", help="Name of the run to be logged.")
# args = parser.parse_args()
# args = vars(args)
# print(args)
_CONFIG = config_flags.DEFINE_config_file("config")
flags.FLAGS(sys.argv)
config = _CONFIG.value
print(config)
# %%
dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_train = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K/")
dst_val = TID2013("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/databases/IQA/TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/databases/IQA/TID/TID2013/", exclude_imgs=[25])

# %%
img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

# %%
img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

# %%
wandb.init(project="PerceptNet_v15",
        #    name="FinalModel_AllFree",
           job_type="training",
           config=config,
           mode="online",
           )
config = config
config

# %%
dst_train_rdy = dst_train.dataset.shuffle(buffer_size=100,
                                      reshuffle_each_iteration=True,
                                      seed=config.SEED)\
                                 .batch(config.BATCH_SIZE, drop_remainder=True)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

# %% [markdown]
# ## Define the model we're going to use
# 
# > It's going to be a very simple model just for demonstration purposes.

# %%
#| exporti
class ChromaFreqOrientGaussianGamma(nn.Module):
    """(1D) Gaussian interaction between gamma_fuencies and orientations optimizing gamma = 1/sigma instead of sigma."""
    use_bias: bool = False
    strides: int = 1
    padding: str = "SAME"
    bias_init: Callable = nn.initializers.zeros_init()
    n_scales: Sequence[int] = jnp.array([4, 2, 2], dtype=jnp.int32)
    n_orientations: Sequence[int] = jnp.array([8, 8, 8], dtype=jnp.int32)

    @nn.compact
    def __call__(self,
                 inputs,
                 fmean,
                 theta_mean,
                 **kwargs,
                 ):

        gamma_f_a = self.param("gamma_f_a",
                             k_array(1/0.4, arr=jnp.array([2.,4.,8.,16.])),
                             (self.n_scales[0],))
        gamma_theta_a = self.param("gamma_theta_a",
                                 nn.initializers.ones_init(),
                                #  (self.n_orientations[0],))
                                 (8,))

        gamma_f_t = self.param("gamma_f_t",
                             k_array(1/0.4, arr=jnp.array([3.,6.])),
                             (self.n_scales[1],))
        gamma_theta_t = self.param("gamma_theta_t",
                                 nn.initializers.ones_init(),
                                #  (self.n_orientations[1],))
                                 (8,))
        
        gamma_f_d = self.param("gamma_f_d",
                             k_array(1/0.4, arr=jnp.array([3.,6.])),
                             (self.n_scales[2],))
        gamma_theta_d = self.param("gamma_theta_d",
                                 nn.initializers.ones_init(),
                                #  (self.n_orientations[2],))
                                 (8,))

        H_cc = self.param("H_cc",
                          nn.initializers.ones_init(),
                          (3,3))

        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (len(fmean),))
        else: bias = 0.
        # n_groups = inputs.shape[-1] // len(fmean)

        ## Repeat gammas
        gamma_f = jnp.concatenate([jnp.tile(f, reps=len(t)) for f,t in zip([gamma_f_a, gamma_f_t, gamma_f_d], [gamma_theta_a, gamma_theta_t, gamma_theta_d])])
        gamma_f = jnp.tile(gamma_f, reps=2)
        gamma_theta = jnp.concatenate([jnp.tile(t, reps=len(f)) for f,t in zip([gamma_f_a, gamma_f_t, gamma_f_d], [gamma_theta_a, gamma_theta_t, gamma_theta_d])])
        gamma_theta = jnp.tile(gamma_theta, reps=2)

        ## Repeating
        cc = jnp.array([0,1,2])
        cc = jnp.repeat(cc, repeats=jnp.array([64,32,32]), total_repeat_length=len(fmean))

        kernel = jax.vmap(self.gaussian, in_axes=(None,None,0,0,0,0,None,0,None,None), out_axes=1)(fmean, theta_mean, fmean, theta_mean, gamma_f, gamma_theta, cc, cc, H_cc, 1)
        kernel = kernel[None,None,:,:]

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv_general_dilated(
                jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
                jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
                (self.strides, self.strides),
                self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gaussian(f, theta, fmean, theta_mean, gamma_f, gamma_theta, c_1, c_2, H_cc, A=1):
        return H_cc[c_1,c_2]*A*jnp.exp(-((gamma_f**2)*(f-fmean)**2)/(2))*jnp.exp(-((gamma_theta**2)*(theta-theta_mean)**2)/(2))

# %%
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
        return self.param("B", nn.initializers.ones_init(), (outputs.shape[-1],))*inputs / (jnp.clip(outputs+bias, a_min=1e-5)**self.epsilon + self.eps)

# %%
class GaborLayerGammaHumanLike_(nn.Module):
    """Parametric Gabor layer with particular initialization."""
    n_scales: Sequence[int] # [A, T, D]
    n_orientations: Sequence[int] # [A, T, D]

    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1

    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    phase = jnp.array([0., jnp.pi/2.])

    normalize_prob: bool = True
    normalize_energy: bool = False
    zero_mean: bool = False
    train_A: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 return_freq=False,
                 return_theta=False,
                 ):
        total_scales = jnp.sum(jnp.array(self.n_scales))
        total_orientations = jnp.sum(jnp.array(self.n_orientations))
        features = jnp.sum(jnp.array([s*o*len(self.phase) for s, o in zip(self.n_scales, self.n_orientations)]))

        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], features))
        freq_a = self.param("freq_a",
                           freq_scales_init(n_scales=self.n_scales[0], fs=self.fs),
                           (self.n_scales[0],))
        gammax_a = self.param("gammax_a",
                           k_array(k=0.4, arr=1/(freq_a**0.8)),
                           (self.n_scales[0],))
        gammay_a = self.param("gammay_a",
                            equal_to(gammax_a*0.8),
                            (self.n_scales[0],))
        theta_a = self.param("theta_a",
                           linspace(start=0, stop=jnp.pi, num=self.n_orientations[0]),
                           (self.n_orientations[0],))
        sigma_theta_a = self.param("sigma_theta_a",
                                  equal_to(theta_a),
                                  (self.n_orientations[0],))

        freq_t = self.param("freq_t",
                           freq_scales_init(n_scales=self.n_scales[1], fs=self.fs),
                           (self.n_scales[1],))
        gammax_t = self.param("gammax_t",
                           k_array(k=0.4, arr=1/(freq_t**0.8)),
                           (self.n_scales[1],))
        gammay_t = self.param("gammay_t",
                            equal_to(gammax_t*0.8),
                            (self.n_scales[1],))
        theta_t = self.param("theta_t",
                           linspace(start=0, stop=jnp.pi, num=self.n_orientations[1]),
                           (self.n_orientations[1],))
        sigma_theta_t = self.param("sigma_theta_t",
                                  equal_to(theta_t),
                                  (self.n_orientations[1],))

        freq_d = self.param("freq_d",
                           freq_scales_init(n_scales=self.n_scales[2], fs=self.fs),
                           (self.n_scales[2],))
        gammax_d = self.param("gammax_d",
                           k_array(k=0.4, arr=1/(freq_d**0.8)),
                           (self.n_scales[2],))
        gammay_d = self.param("gammay_d",
                            equal_to(gammax_d*0.8),
                            (self.n_scales[2],))
        theta_d = self.param("theta_d",
                           linspace(start=0, stop=jnp.pi, num=self.n_orientations[2]),
                           (self.n_orientations[2],))
        sigma_theta_d = self.param("sigma_theta_d",
                                  equal_to(theta_d),
                                  (self.n_orientations[2],))

        # A = self.param("A",
        #                nn.initializers.ones_init(),
        #                (inputs.shape[-1], features)) if self.train_A else jnp.ones(shape=(inputs.shape[-1], features))
        A = self.param("A",
                       nn.initializers.ones_init(),
                       (inputs.shape[-1], 128))
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (features,))
        else: bias = 0.

        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            ## A
            kernel_a = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,None,None,None,None,None,None,None), out_axes=0)
            kernel_a = jax.vmap(kernel_a, in_axes=(None,None,None,None,None,None,None,0,0,None,None,None,None,None), out_axes=0)
            kernel_a = jax.vmap(kernel_a, in_axes=(None,None,None,None,None,None,None,None,None,0,None,None,None,None), out_axes=0)(x, y, self.xmean, self.ymean, gammax_a, gammay_a, freq_a, theta_a, sigma_theta_a, self.phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean)
            kernel_a = rearrange(kernel_a, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)")
            kernel_a = repeat(kernel_a, "kx ky c_out -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=kernel_a.shape[-1])

            ## T
            kernel_t = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,None,None,None,None,None,None,None), out_axes=0)
            kernel_t = jax.vmap(kernel_t, in_axes=(None,None,None,None,None,None,None,0,0,None,None,None,None,None), out_axes=0)
            kernel_t = jax.vmap(kernel_t, in_axes=(None,None,None,None,None,None,None,None,None,0,None,None,None,None), out_axes=0)(x, y, self.xmean, self.ymean, gammax_t, gammay_t, freq_t, theta_t, sigma_theta_t, self.phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean)
            kernel_t = rearrange(kernel_t, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)")
            kernel_t = repeat(kernel_t, "kx ky c_out -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=kernel_t.shape[-1])

            ## D
            kernel_d = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,None,None,None,None,None,None,None), out_axes=0)
            kernel_d = jax.vmap(kernel_d, in_axes=(None,None,None,None,None,None,None,0,0,None,None,None,None,None), out_axes=0)
            kernel_d = jax.vmap(kernel_d, in_axes=(None,None,None,None,None,None,None,None,None,0,None,None,None,None), out_axes=0)(x, y, self.xmean, self.ymean, gammax_d, gammay_d, freq_d, theta_d, sigma_theta_d, self.phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean)
            kernel_d = rearrange(kernel_d, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)")
            kernel_d = repeat(kernel_d, "kx ky c_out -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=kernel_d.shape[-1])

            ## Concat all of them
            kernel = jnp.concatenate([kernel_a, kernel_t, kernel_d], axis=-1)
            kernel = kernel * A[None,None,:,:]
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        fmean = jnp.concatenate([jnp.tile(f, reps=len(t)) for f,t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])])
        fmean = jnp.tile(fmean, reps=2)
        theta_mean = jnp.concatenate([jnp.tile(t, reps=len(f)) for f,t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])])
        theta_mean = jnp.tile(theta_mean, reps=2)

        if not had_batch: outputs = outputs[0]
        if return_freq and return_theta:
            return outputs + bias, fmean, theta_mean 
        elif return_freq and not return_theta:
            return outputs + bias, fmean
        elif not return_freq and return_theta:
            return outputs + bias, theta_mean
        else:
            return outputs + bias

    @staticmethod
    def gabor(x, y, xmean, ymean, gammax, gammay, freq, theta, sigma_theta, phase, A=1, normalize_prob=True, normalize_energy=False, zero_mean=False):
        x, y = x-xmean, y-ymean
        ## Obtain the normalization coeficient
        gamma_vector = jnp.array([gammax, gammay])
        inv_cov_matrix = jnp.diag(gamma_vector)**2
        # det_cov_matrix = 1/jnp.linalg.det(cov_matrix)
        # # A_norm = 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)) if normalize_prob else 1.
        # A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)), 1.)
        A_norm = 1.
        
        ## Rotate the sinusoid
        rotation_matrix = jnp.array([[jnp.cos(sigma_theta), -jnp.sin(sigma_theta)],
                                     [jnp.sin(sigma_theta), jnp.cos(sigma_theta)]])
        rotated_covariance = rotation_matrix @ inv_cov_matrix @ jnp.transpose(rotation_matrix)
        x_r_1 = rotated_covariance[0,0] * x + rotated_covariance[0,1] * y
        y_r_1 = rotated_covariance[1,0] * x + rotated_covariance[1,1] * y
        distance = x * x_r_1 + y * y_r_1
        g = A_norm*jnp.exp(-distance/2) * jnp.cos(2*jnp.pi*freq*(x*jnp.cos(theta)+y*jnp.sin(theta)) + phase)
        g = jnp.where(zero_mean, g - g.mean(), g)
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        return A*g/E_norm

    def return_kernel(self, params, c_in=3):
        x, y = self.generate_dominion()
        sigmax, sigmay = jnp.exp(params["sigmax"]), jnp.exp(params["sigmay"])
        kernel = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,None,None,None,None,None,None,None), out_axes=0)
        kernel = jax.vmap(kernel, in_axes=(None,None,None,None,None,None,0,None,None,None,None,None,None), out_axes=0)
        kernel = jax.vmap(kernel, in_axes=(None,None,None,None,None,None,None,0,0,0,None,None,None), out_axes=0)(x, y, self.xmean, self.ymean, params["sigmax"], params["sigmay"], params["freq"], params["theta"], params["sigma_theta"], self.phase, 1, self.normalize_prob, self.normalize_energy)
        # kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
        kernel = rearrange(kernel, "rots fs sigmas kx ky -> kx ky (rots fs sigmas)")
        kernel = repeat(kernel, "kx ky c_out -> kx ky c_in c_out", c_in=c_in, c_out=kernel.shape[-1])
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size), jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size))

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
        if config.PARAM_CS:
            outputs = CenterSurroundLogSigmaK(features=3, kernel_size=config.CS_KERNEL_SIZE, fs=21, use_bias=False, padding="VALID")(outputs, **kwargs)
        else:
            outputs = nn.Conv(features=3, kernel_size=(config.CS_KERNEL_SIZE,config.CS_KERNEL_SIZE), use_bias=False, padding="VALID")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))

        ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
        ### fs = 32 / kernel_size = (11,11) -> 0.32 > 0.02 --> OK!
        ## TO-DO: - Spatial Gaussian Kernel (0.02 deg) -> fs = 64/2 & 0.02*64/2 = sigma (px) = 0.69
        if config.PARAM_DN_CS:
            outputs = GDNGaussian(kernel_size=config.GDNGAUSSIAN_KERNEL_SIZE, apply_independently=True, fs=32, padding="symmetric", normalize_prob=config.NORMALIZE_PROB, normalize_energy=config.NORMALIZE_ENERGY)(outputs, **kwargs)

        else:
            # outputs = pad_same_from_kernel_size(outputs, kernel_size=config.GDNGAUSSIAN_KERNEL_SIZE, mode="symmetric")
            outputs = GDN(kernel_size=(config.GDNGAUSSIAN_KERNEL_SIZE,config.GDNGAUSSIAN_KERNEL_SIZE), apply_independently=True, padding="SAME")(outputs)
        ## GaborLayer per channel with GDN mixing only same-origin-channel information
        ### [Gaussian] sigma = 0.2 (deg) fs = 32 / kernel_size = (21,21) -> 21/32 = 0.66 --> OK!
        outputs = pad_same_from_kernel_size(outputs, kernel_size=config.GABOR_KERNEL_SIZE, mode="symmetric")
        # outputs, fmean, theta_mean = GaborLayerGamma_(n_scales=4+2+2, n_orientations=8*3, kernel_size=config.GABOR_KERNEL_SIZE, fs=32, xmean=config.GABOR_KERNEL_SIZE/32/2, ymean=config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=config.NORMALIZE_PROB, normalize_energy=config.NORMALIZE_ENERGY, zero_mean=config.ZERO_MEAN, use_bias=config.USE_BIAS, train_A=config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        if config.PARAM_GABOR:
            outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(n_scales=[4,2,2], n_orientations=[8,8,8], kernel_size=config.GABOR_KERNEL_SIZE, fs=32, xmean=config.GABOR_KERNEL_SIZE/32/2, ymean=config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=config.NORMALIZE_PROB, normalize_energy=config.NORMALIZE_ENERGY, zero_mean=config.ZERO_MEAN, use_bias=config.USE_BIAS, train_A=config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        ## Final GDN mixing Gabor information (?)
            outputs = GDNSpatioChromaFreqOrient(kernel_size=21, strides=1, padding="symmetric", fs=32, apply_independently=False)(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)
        else:
            outputs = pad_same_from_kernel_size(outputs, kernel_size=config.GABOR_KERNEL_SIZE, mode="symmetric")
            outputs = nn.Conv(features=128, kernel_size=(config.GABOR_KERNEL_SIZE,config.GABOR_KERNEL_SIZE), padding="VALID", use_bias=False)(outputs)
            # outputs = pad_same_from_kernel_size(outputs, kernel_size=21, mode="symmetric")
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
# ## Defining the training step
# 
# > We want to write a function that takes the `TrainState` and a batch of data can performs an optimization step.

# %%
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

# %%
from functools import partial

# %%
@partial(jax.jit, static_argnums=2)
def train_step(state, batch, return_grads=False):
    """Train for a single step."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        img_pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
        img_dist_pred, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)

        ## Calculate the distance
        dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)
        
        ## Calculate pearson correlation
        return pearson_correlation(dist, mos), updated_state
    
    (loss, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    state = state.replace(state=updated_state)
    if return_grads: return state, grads
    else: return state

# %% [markdown]
# In their example, they don't calculate the metrics at the same time. I think it is kind of a waste because it means having to perform a new forward pass, but we'll follow as of now. Let's define a function to perform metric calculation:

# %%
@jax.jit
def compute_metrics(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        img_pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=False)
        img_dist_pred, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=False)

        ## Calculate the distance
        dist = ((img_pred - img_dist_pred)**2).sum(axis=(1,2,3))**(1/2)
        
        ## Calculate pearson correlation
        return pearson_correlation(dist, mos)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params))
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

# %% [markdown]
# ## Train the model!

# %%
if hasattr(config, "LEARNING_RATE"): tx = optax.adam(config.LEARNING_RATE)
else: tx = optax.adam(config.PEAK_LR)
state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# %%
state.params.keys()

# %%
pred, _ = state.apply_fn({"params": state.params, **state.state}, jnp.ones(shape=(1,384,512,3)), train=True, mutable=list(state.state.keys()))
state = state.replace(state=_)

# %%
try:
    state.state["precalc_filter"]["GaborLayerGammaHumanLike__0"]["kernel"].shape
except:
    pass
# %%
import matplotlib.pyplot as plt

# %%
def plot_filters(kernel):
    kx, ky, cin, cout = kernel.shape
    fig, axes = plt.subplots(cin, cout, squeeze=False)
    for i in range(cin):
        for j in range(cout):
            axes[i,j].imshow(kernel[:,:,i,j])
            axes[i,j].axis("off")
    plt.show()

# %%
__ = jax.tree_util.tree_map(lambda x: plot_filters(x), _)

# %%
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
print(trainable_tree)

# %%
if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(learning_rate=config.LEARNING_RATE)
else:
    steps_per_epoch = len(dst_train_rdy)
    epochs = 500
    schedule_lr = optax.warmup_cosine_decay_schedule(init_value=config.INITIAL_LR, peak_value=config.PEAK_LR, end_value=config.END_LR, warmup_steps=steps_per_epoch*config.WARMUP_EPOCHS, decay_steps=steps_per_epoch*config.EPOCHS)
    tx = optax.adam(learning_rate=schedule_lr)

optimizers = {
    "trainable": tx,
    "non_trainable": optax.set_to_zero(),
}

# %%
tx = optax.multi_transform(optimizers, trainable_tree)

# %%
state = create_train_state(PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
print(param_count, trainable_param_count)

# %%
wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

# %%
state = state.replace(params=unfreeze(state.params))

# ## DN 0 (Gamma)
if config.USE_GAMMA:
    state.params["GDNGamma_0"]["bias"] = jnp.ones_like(state.params["GDNGamma_0"]["bias"])*0.1
    state.params["GDNGamma_0"]["kernel"] = jnp.ones_like(state.params["GDNGamma_0"]["kernel"])*0.5
else:
    state.params["GDN_0"]["Conv_0"]["bias"] = jnp.ones_like(state.params["GDN_0"]["Conv_0"]["bias"])*0.1
    state.params["GDN_0"]["Conv_0"]["kernel"] = jnp.ones_like(state.params["GDN_0"]["Conv_0"]["kernel"])*0.5

## Opponent color channel transformation
if config.INIT_JH:
    state.params["Color"]["kernel"] = jnp.array([[39.0454,30.1207,14.27948],
                                                  [115.8404,-63.3502,41.26816],
                                                  [16.3118,30.2934,-61.51888]])[None,None,:,:]/163.5217

## Center Surround
if config.PARAM_CS:
    state.params["CenterSurroundLogSigmaK_0"]["logsigma"] = jnp.array([-1.9,-1.9,-1.9,
                                                                   -1.76,-1.76,-1.76,
                                                                   -1.76,-1.76,-1.76])
    state.params["CenterSurroundLogSigmaK_0"]["K"] = jnp.array([1.1,1.1,1.1,
                                                            5.0,5.0,5.0,
                                                            5.0,5.0,5.0])
    state.params["CenterSurroundLogSigmaK_0"]["A"] = jnp.array([1.,0.,0.,
                                                            0.,1.,0.,
                                                            0.,0.,1.])

## GDNGaussian
if config.PARAM_DN_CS:
    state.params["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.ones_like(state.params["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"])*(1./0.04)
    state.params["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"] = jnp.ones_like(state.params["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"])*0.1

## Gabor
if config.PARAM_GABOR and config.INIT_GABOR:
    state.params["GaborLayerGammaHumanLike__0"]["freq_a"] = jnp.array([2.,4.,8.,16.]) 
    state.params["GaborLayerGammaHumanLike__0"]["freq_t"] = jnp.array([3.,6.]) 
    state.params["GaborLayerGammaHumanLike__0"]["freq_d"] = jnp.array([3.,6.])

    state.params["GaborLayerGammaHumanLike__0"]["gammax_a"] = state.params["GaborLayerGammaHumanLike__0"]["freq_a"]**0.9
    state.params["GaborLayerGammaHumanLike__0"]["gammay_a"] = 0.8*state.params["GaborLayerGammaHumanLike__0"]["gammax_a"]

    state.params["GaborLayerGammaHumanLike__0"]["gammax_t"] = state.params["GaborLayerGammaHumanLike__0"]["freq_t"]**0.9
    state.params["GaborLayerGammaHumanLike__0"]["gammay_t"] = 0.8*state.params["GaborLayerGammaHumanLike__0"]["gammax_t"]

    state.params["GaborLayerGammaHumanLike__0"]["gammax_d"] = state.params["GaborLayerGammaHumanLike__0"]["freq_d"]**0.9
    state.params["GaborLayerGammaHumanLike__0"]["gammay_d"] = 0.8*state.params["GaborLayerGammaHumanLike__0"]["gammax_d"]
    # state.params["GaborLayerGammaHumanLike__0"]["theta_a"] = jnp.tile(jnp.linspace(0., jnp.pi, num=16), reps=128//16)
    # state.params["GaborLayerGammaHumanLike__0"]["sigma_theta_a"] = state.params["GaborLayerGammaHumanLike__0"]["theta_a"]
    # state.params["GaborLayerGammaHumanLike__0"]["phase_a"] = jnp.repeat(jnp.array([0., 90.]), repeats=64)    

    A_a = jnp.zeros(shape=(3,64), dtype=jnp.float32)
    A_a = A_a.at[0,:].set(1.)
    A_t = jnp.zeros(shape=(3,32), dtype=jnp.float32)
    A_t = A_t.at[1,:].set(1.)
    A_d = jnp.zeros(shape=(3,32), dtype=jnp.float32)
    A_d = A_d.at[2,:].set(1.)
    state.params["GaborLayerGammaHumanLike__0"]["A"] = jnp.concatenate([A_a, A_t, A_d], axis=-1)

## GDNSpatioChromaFreqOrient
# state.params["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.ones_like(state.params["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"])*(1./0.1)
# state.params["GDNSpatioChromaFreqOrient_0"]["OrientGaussianGamma_0"]["gamma"] = jnp.ones_like(state.params["GDNSpatioChromaFreqOrient_0"]["OrientGaussianGamma_0"]["gamma"])*(1/20)
# state.params["GDNSpatioChromaFreqOrient_0"]["bias"] = jnp.tile(jnp.array([0.001, 0.002, 0.0035, 0.01])/100, reps=config.N_ORIENTATIONS*2)
if config.PARAM_GABOR:
    state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"] = jnp.ones_like(state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"])*(1/20)
    state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"] = jnp.ones_like(state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"])*(1/20)
    state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"] = jnp.ones_like(state.params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"])*(1/20)


state = state.replace(params=freeze(state.params))

# %%
pred, _ = state.apply_fn({"params": state.params, **state.state}, jnp.ones(shape=(1,384,512,3)), train=True, mutable=list(state.state.keys()))
state = state.replace(state=_)

# %%
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-0"), state, save_args=save_args, force=True) # force=True means allow overwritting.

# %%
metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

# %%
batch = next(iter(dst_train_rdy.as_numpy_iterator()))

# %%
from functools import partial

# %%
@jax.jit
def forward(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False)

# %%
@jax.jit
def forward_intermediates(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False, capture_intermediates=True)

# %%
# %%time
outputs = forward(state, batch[0])
outputs.shape

# %%
# %%time
s1, grads = train_step(state, batch, return_grads=True)

# %%
# jax.config.update("jax_debug_nans", True)

# %%
def filter_extra(extra):
    def filter_intermediates(path, x):
        path = "/".join(path)
        if "Gabor" in path:
            return (x[0][0],)
        else: 
            return x
    extra = unfreeze(extra)
    extra["intermediates"] = flax.traverse_util.path_aware_map(filter_intermediates, extra["intermediates"])
    return freeze(extra)

# %%
# %%time
step = 0
for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state, grads = train_step(state, batch, return_grads=True)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        state = state.replace(params=clip_param(state.params, "A", a_min=0))
        state = state.replace(params=clip_param(state.params, "K", a_min=1+1e-5))
        wandb.log({f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()}, commit=False)
        step += 1
        # state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)
    
    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        # break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())
    
    ## Obtain activations of last validation batch
    _, extra = forward_intermediates(state, batch[0])
    extra = filter_extra(extra) ## Needed because the Gabor layer has multiple outputs
    
    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-best"), state, save_args=save_args, force=True) # force=True means allow overwritting.
    # orbax_checkpointer.save(os.path.join(wandb.run.dir, f"model-{epoch+1}"), state, save_args=save_args, force=False) # force=True means allow overwritting.

    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()}, commit=False)
    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(extra["intermediates"]).items()}, commit=False)
    if hasattr(config, "LEARNING_RATE"):
        wandb.log({"epoch": epoch+1, "learning_rate": config.LEARNING_RATE, **{name:values[-1] for name, values in metrics_history.items()}})
    else:
        wandb.log({"epoch": epoch+1, "learning_rate": schedule_lr(step), **{name:values[-1] for name, values in metrics_history.items()}})
    print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}')
    # break


# %% [markdown]
# Save the final model as well in case we want to keep training from it or whatever:

# %%
orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args)

# %%
wandb.finish()

