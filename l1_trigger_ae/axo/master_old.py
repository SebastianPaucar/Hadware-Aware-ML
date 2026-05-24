import sys

import numpy as np
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import h5py

#from qkeras.utils import _add_supported_quantized_objects
#from qkeras import quantized_bits
#co = {}; _add_supported_quantized_objects(co)

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

K = tf.keras.backend

from . import data_util
from . import losses
from . import models
from . import optim
from . import metric
from . import losses
from . import callbacks as axo_callbacks
from . import utilities

from .callbacks import BitwidthLogger
from .callbacks import PruningMetrics

import gc
import re
import pprint
import json

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler
tf.config.run_functions_eagerly(True)

import mplhep as hep
from tqdm.auto import tqdm
import argparse
import yaml
import os

from keras.models import load_model

# ===== Import your custom layers =====
from HGQ import HDense, HQuantize   # make sure this path matches where HGQ.py is

# ===== Load the model =====
#encoder_to_mean = load_model(
#    "encoder_to_mean_model.h5",
#    custom_objects={"HDense": HDense, "HQuantize": HQuantize}
#)

from HGQ import ResetMinMax, FreeBOPs



def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def main(slave = {},data_creation=False, train_model=False, inference = False):
    
    ########################################################################################################
    # Dictionary merging and config checking
    ########################################################################################################
    module_dir = os.path.dirname(__file__)
    
    master = yaml.load(open(os.path.join(module_dir,"utilities/config.yml"),"r"), Loader=yaml.Loader)

    if utilities.check_compartibility(master=master,slave=slave) == 1:
        print("Configurations are compartible")
        print("Generating new config ....")
        daughter = utilities.merge_dict(master=master,slave=slave)
    else:
        print("Smoke test failed !!! check the dictionary and the documentations")
        return 0
    config = daughter.copy() ## Setting Daughter as default
    ########################################################################################################
    # Reproducibility
    ########################################################################################################
    seed = config["determinism"]["global_seed"]
    if config["determinism"]["python_determinism"]:
        os.environ["PYTHONHASHSEED"] = str(seed)
    if config["determinism"]["numpy_determinism"]:
        import numpy as np
        np.random.seed(seed)
    if config["determinism"]["tf_op_determinism"]:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()

    if data_creation == True:
        ########################################################################################################
        # Data creation
        ########################################################################################################
        processed_data_path = config["data_config"]["Processed_data_path"]
        if os.path.isfile(processed_data_path):
            print("File already exists, checking if the config match")
            f = h5py.File(config["data_config"]["Processed_data_path"],"r")
            exisitng_ser_config = f.attrs["config"]
            f.close()
            present_ser_config = json.dumps(config["data_config"])
            if exisitng_ser_config == present_ser_config:
                print("Configs match, skipping")
            else:
                print("[WARNINIG]:CONFIG DO NOT MATCH, OVERWRITING!!!")
                data_util.data.get_data(config_master=config["data_config"])
        else:
            print("File does not exist, creating data file")
            data_util.data.get_data(config_master=config["data_config"])
    if train_model == True:
        ########################################################################################################
        # Data reading
        ########################################################################################################
        f = h5py.File(config["data_config"]["Processed_data_path"],"r")
        x_train = f["Background_data"]["Train"]["DATA"][:]
        x_test = f["Background_data"]["Test"]["DATA"][:]
        scale = f["Normalisation"]["norm_scale"][:]
        bias = f["Normalisation"]["norm_bias"][:]
        f.close()
        x_train = np.reshape(x_train,(x_train.shape[0],-1))
        x_test = np.reshape(x_test,(x_test.shape[0],-1))
        ########################################################################################################
        # Loss Setup
        ########################################################################################################
        loss_name = config["train"]["common"]["reconstruction_loss"].split("_loss")[0] # This is for lousy users
        constituents = config["data_config"]["Read_configs"]["BACKGROUND"]["constituents"]
        compute_loss = getattr(losses, f"{loss_name}_loss")
        loss_reco = compute_loss(norm_scales=scale,
                                 norm_biases=bias,
                                 mask=constituents,
                                 name="Reco_loss"
                                )
        loss_kld = losses.kld()
        print("Configured reconstruction loss:", config["train"]["common"]["reconstruction_loss"])
        print("Derived loss_name:", loss_name)
        print("Loss callable found:", compute_loss)
        print("Loss callable type:", type(compute_loss))

        ########################################################################################################
        # Model Setup
        ########################################################################################################
        config_to_vae = {**config["model"].copy(), **config["train"]["VAE_config"].copy()} ## <--- This can be further generalised when we have more than one models in the production.
        vae = models.VariationalAutoEncoderHGQ(config = config_to_vae,
                                            reco_loss = loss_reco,
                                            kld_loss= loss_kld) ## <--- This can be further generalised when we have more than one models in the production.
        ########################################################################################################
        # Optimiser Setup
        ########################################################################################################
        optim_config = config["train"]['common']['optimiser_config']
        optim_name = optim_config['optmiser']
        try:
            compute_optim = getattr(optim, optim_name)
        except AttributeError:
                print("Optimizer Not found in AXO, looking in TF")
                compute_optim = getattr(tf.keras.optimizers, optim_name)
        allowed_params_for_optim = utilities.allowed_params(compute_optim)
        allowed_keys = list(set(optim_config.keys()).intersection(set(allowed_params_for_optim)))
        config_to_optim = {k: v for k, v in optim_config.items() if k in allowed_keys}
        opt = compute_optim(**config_to_optim)
    ########################################################################################################
    # Model Compilation
    ########################################################################################################
        vae.compile(optimizer=opt)
    ########################################################################################################
    # Callback Setup
    ########################################################################################################
    ### For now we will only deal with lr_scheduler more can be added as per need :>
        callbacks = []
        callback_config = config["callback"]
    # Checking for LR Scheduler
        try:
            lrsc_config = callback_config["lr_schedule"]
        except KeyError:
            print("Learning rate scheduler not found; training with constant learning rate.")
        else:
            lrsc_name = lrsc_config["name"]
            lrsc_config = lrsc_config["config"]
            try:
                compute_lrsc = getattr(axo_callbacks.lr_scheduler, lrsc_name)
            except AttributeError:
                print("Scheduler not found in AXO, looking in TF callbacks")
                compute_lrsc = getattr(tf.keras.callbacks, lrsc_name)

        # Implementing the lrsc
            allowed_params_for_lrsc = utilities.allowed_params(compute_lrsc)
            allowed_keys = list(set(lrsc_config.keys()).intersection(set(allowed_params_for_lrsc)))
            config_to_lrsc = {k: v for k, v in lrsc_config.items() if k in allowed_keys}

        # Wrap the scheduler function to ensure it returns a float
            def wrapped_lr_schedule(epoch, lr):
                new_lr = compute_lrsc(**config_to_lrsc)(epoch, lr)  # Call the original scheduler function
                return float(new_lr)  # Ensure the output is a float
        # Create the LearningRateScheduler with the wrapped function
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(wrapped_lr_schedule, verbose=1)
            callbacks.append(lr_scheduler)
            callbacks.append(BitwidthLogger())
            callbacks.append(PruningMetrics(prune_threshold=0.5))
    ########################################################################################################
    # Model Training
    ########################################################################################################
        history = vae.fit(x_train,x_train,
                    callbacks=[callbacks, ResetMinMax(), FreeBOPs()],
                    batch_size=config["train"]["common"]["batch_size"],
                    epochs=config["train"]["common"]["n_epochs"],
                    validation_split=0.1,
                    shuffle=True,
                    verbose = 2
                #callbacks = callbacks.callbacks
                )
    ########################################################################################################
    # Model Trimming
    ########################################################################################################
 #       model = tf.keras.Sequential(vae.encoder.layers[:-2])
       # model.save("trimmed_encoder_savedmodel.keras")
       # print("Trimmed encoder guardado como 'trimmed_encoder.h5'")
        model = Model(
            inputs=vae.encoder.input,
            outputs=vae.encoder.get_layer("latent_mean").output
        )

        print("\n[INFO] Encoder-to-mean model summary:")
        model.summary()
        model.save(config["trimmed_encoder"])  # cut encoder
        print("\n[INFO] Model saved")

    ########################################################################################################
    # Score plots and result generation
    ########################################################################################################
        config_thres = config["threshold"]
        config_thres["data_path"] = config["data_config"]["Processed_data_path"]
    #    config_thres["QAT"] = config["model"]["encoder_config"]["QAT"]
        additional_metrics = config["additional_metrics"]
        axo_man = metric.axo_threshold_manager(model=model,config=config_thres, additional_metrics=additional_metrics)
        dist_plot = metric.distribution_plots(model = model, config=config_thres)
    ########################################################################################################
    # Storage and report generation
    ########################################################################################################
        utilities.store_axo(config = config,
                            model = vae,
                            model_trim = model,
                            axo_man = axo_man,
                            dist_plot = dist_plot,
                            history_dict = history.history
                           )
    
        threshold_dict = utilities.retrieve.get_threshold_dict(config["store"]["lite_path"])
        dict_axo = utilities.retrieve.get_axo_score_dataframes(config["store"]["lite_path"])
        histogram_dict = utilities.retrieve.get_histogram_dict(config["store"]["lite_path"])
        history_dict = utilities.retrieve.get_history_dict(config["store"]["lite_path"])
    
        report_config = config["report"]
        if report_config["html_report"]["generate"] == True or report_config["pdf_report"]["generate"]:
            print("Report generation flag found !!")
            html_path = report_config["html_report"]["path"]
            pdf_path = report_config["pdf_report"]["path"]
            utilities.generate_axolotl_html_report(config = config,
                                                   dict_axo = dict_axo,
                                                   histogram_dict = histogram_dict,
                                                   threshold_dict = threshold_dict,
                                                   history_dict = history_dict,
                                                   output_file = html_path,
                                                   pdf_file=pdf_path)
    
    if inference == True:

        trimmed_encoder = load_model(
            config["trimmed_encoder"],
            custom_objects={"HDense": HDense, "HQuantize": HQuantize}
        )

        print("[INFO] Model loaded successfully.")
        trimmed_encoder.summary()

  #      if config["model"]["encoder_config"]["QAT"]:
   #         trimmed_encoder = load_qmodel(config["inference"]["trimmed_encoder_path"], compile=False)
    #    else:
     #       trimmed_encoder = tf.keras.models.load_model(config["inference"]["trimmed_encoder_path"], compile=False)
      #  trimmed_encoder.summary()
        config_thres = config["threshold"]
       # config_thres["QAT"] = config["model"]["encoder_config"]["QAT"]
        config_thres["data_path"] = config["data_config"]["Processed_data_path"]
        config_thres["inference"] = config["inference"]
        axo_man = metric.axo_threshold_manager(model=trimmed_encoder,config=config_thres,inference=inference)
        raw_score_dict, threshold_dict = axo_man.get_raw_dict()
        print(raw_score_dict)
        print(threshold_dict)
    print("Run completing exiting ...")

