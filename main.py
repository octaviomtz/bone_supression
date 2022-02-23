#%%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers import BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils.misc import load_data
from utils.figures import xray_overview, fig_loss, fig_prediction_xray
from utils.models import autoencoder

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):

    source, target = load_data(cfg.n_images)
    len(source), source[0].shape
    xray_overview(source, target)
    img_shape = (cfg.height, cfg.width, cfg.channels)

    source = np.array(source).reshape(-1, cfg.height, cfg.width, cfg.channels)
    target = np.array(target).reshape(-1, cfg.height, cfg.width, cfg.channels)

    source_train, source_test, target_train, target_test = train_test_split(source, target,
                                                                            test_size=0.20,
                                                                            random_state=1)
    print(source_train.shape, source_test.shape, target_train.shape, target_test.shape)

    input_img = Input(shape = img_shape)
    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    print(autoencoder.summary())

    #%%
    autoencoder_train = autoencoder.fit(source_train, target_train,
                                        epochs = cfg.n_epoch,
                                        batch_size = cfg.n_batch,
                                        verbose = 1,
                                        validation_data = (source_test, target_test))

    interval_epochs = (np.linspace(0, cfg.n_epoch, 5)//1).astype(int)
    for e in interval_epochs:
        print("epoch = {}\tLoss = {:.5f}\tValidation_Loss = {:.5f}".format(e+1,autoencoder_train.history['loss'][e],autoencoder_train.history['val_loss'][e]))                                 

    fig_loss(autoencoder_train, cfg.n_epoch)

    # prediction on validation set
    pred = autoencoder.predict(source_test)
    print(len(pred))

    fig_prediction_xray(source_test, pred, target_test)
