import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from GDN_Jorge import GDN as GDNJ

class GDNWeightWatcherWandb(tf.keras.callbacks.Callback):
    """
    Records the GDN filters after each epoch as an image in Wandb.
    """

    def __init__(self):
        super(GDNWeightWatcherWandb, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, GDNJ):
                for weight in layer.weights:
                    wandb.log({f'{layer.name}.{weight.name}': wandb.Histogram(weight)})

class MOSScatterPlot(tf.keras.callbacks.Callback):
    """
    Represents a scatter plot of the predicted MOS vs the real MOS with
    the corresponding Pearson Correlation Coefficient.
    """
    def __init__(self, train_dataset, test_dataset, freq=1):
        super(MOSScatterPlot, self).__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.freq == 0:
            train_moss, train_moss_pred = self._obtain_preds(self.train_dataset)
            test_moss, test_moss_pred = self._obtain_preds(self.test_dataset)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
            ax[0].scatter(train_moss, train_moss_pred)
            ax[0].set_title(f'Train | Epoch: {epoch} | Pearson: {self.model.correlation_loss(train_moss, train_moss_pred):.2f}')
            ax[0].set_xlabel('MOS')
            ax[0].set_ylabel('MOS_Pred')
            ax[1].scatter(test_moss, test_moss_pred)
            ax[1].set_title(f'Test | Epoch: {epoch} | Pearson: {self.model.correlation_loss(test_moss, test_moss_pred):.2f}')
            ax[1].set_xlabel('MOS')
            ax[1].set_ylabel('MOS_Pred'

            wandb.log({'MOS_Scatter':wandb.Image(fig)})
    
    def on_train_end(self, logs=None):
        self.on_epoch_end(self.freq,logs)

    def _obtain_preds(self, dataset):
        moss = []
        moss_pred = []
        for img, dist, mos in dataset:
            moss.extend(mos.numpy().tolist())
            moss_pred.extend(self.model.predict((img,dist)).squeeze().tolist())
        return moss, moss_pred