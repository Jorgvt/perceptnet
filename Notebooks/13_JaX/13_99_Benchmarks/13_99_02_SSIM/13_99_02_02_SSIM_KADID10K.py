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

import torch
from piq import SSIMLoss as SSIM
from ml_collections import ConfigDict

import wandb

from iqadatasets.datasets import *

# %%
# dst_train = KADID10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/KADID10K/", exclude_imgs=[25])
# dst_val = KADID10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/KADID10K/", exclude_imgs=[25])
# dst_train = KADID10K("/media/disk/databases/BBDD_video_image/Image_Quality//TID/KADID10K/", exclude_imgs=[25])
# dst_val = KADID10K("/media/disk/databases/BBDD_video_image/Image_Quality//TID/KADID10K/", exclude_imgs=[25])
# dst = KADIK10K("/media/disk/databases/BBDD_video_image/Image_Quality/KADIK10K/")
# dst = PIPAL("/media/disk/databases/BBDD_video_image/Image_Quality/PIPAL/")
dst = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA//KADIK10K/")

# %%
img, img_dist, mos = next(iter(dst.dataset))
img.shape, img_dist.shape, mos.shape

# %%

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = None
print(device)

# %%
model = SSIM(reduction="none")
model = model.to(device)

# %%
config = ConfigDict({
    "BATCH_SIZE": 32,
})

# %%
wandb.init(project="PerceptNet_JaX_Eval",
           name="SSIM",
           job_type="evaluate",
           mode="online",
           )
print(config)

# %%
dst_rdy = dst.dataset.batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

# %%
def compute_distance(model, batch, device=None):
    img, img_dist, mos = batch
    img = img.transpose(0,3,1,2)
    img_dist = img_dist.transpose(0,3,1,2)
    img, img_dist = torch.Tensor(img), torch.Tensor(img_dist)
    img, img_dist = img.to(device), img_dist.to(device)

    with torch.no_grad():
        dist = model(img, img_dist)

    return dist

# %%
metrics_history = {
    "distance": [],
    "mos": [],
}

# %%

for batch in tqdm(dst_rdy.as_numpy_iterator()):
    img, img_dist, mos = batch
    distance = compute_distance(model=model, batch=batch, device=device)
    metrics_history["distance"].extend(distance.cpu().numpy())
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
