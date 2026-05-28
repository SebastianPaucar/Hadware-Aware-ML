import numpy as np
import tensorflow as tf
import h5py
from keras.models import Model
from hgq.utils.sugar import FreeEBOPs

from .. import losses
from .. import models
from .. import optim
from .. import metric
from .. import callbacks as axo_callbacks
from .. import utilities
from ..callbacks import BitwidthLogger, HardwareLogger


def run(config):
    x_train, x_test, scale, bias = _load_data(config)
    loss_reco, loss_kld = _setup_losses(config, scale, bias)
    vae = _setup_model(config, loss_reco, loss_kld)
    opt = _setup_optimizer(config)
    vae.compile(optimizer=opt, jit_compile=True)
    callbacks = _setup_callbacks(config)

    history = vae.fit(
        x_train,
        x_train,
        callbacks=[*callbacks, FreeEBOPs()],
        batch_size=config["train"]["common"]["batch_size"],
        epochs=config["train"]["common"]["n_epochs"],
        validation_split=0.1,
        shuffle=True,
        verbose=2
    )

    model = _trim_encoder(vae, config)
    _generate_results(config, vae, model, history)


def _load_data(config):
    f = h5py.File(config["data_config"]["Processed_data_path"], "r")
    x_train = f["Background_data"]["Train"]["DATA"][:]
    x_test = f["Background_data"]["Test"]["DATA"][:]
    scale = f["Normalisation"]["norm_scale"][:]
    bias = f["Normalisation"]["norm_bias"][:]
    f.close()
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    return x_train, x_test, scale, bias


def _setup_losses(config, scale, bias):
    loss_name = config["train"]["common"]["reconstruction_loss"].split("_loss")[0]
    constituents = config["data_config"]["Read_configs"]["BACKGROUND"]["constituents"]
    compute_loss = getattr(losses, f"{loss_name}_loss")
    loss_reco = compute_loss(
        norm_scales=scale,
        norm_biases=bias,
        mask=constituents,
        name="Reco_loss"
    )
    loss_kld = losses.kld()
    print("Configured reconstruction loss:", config["train"]["common"]["reconstruction_loss"])
    print("Loss callable found:", compute_loss)
    return loss_reco, loss_kld


def _setup_model(config, loss_reco, loss_kld):
    config_to_vae = {**config["model"].copy(), **config["train"]["VAE_config"].copy()}
    return models.VariationalAutoEncoderHGQ2(
        config=config_to_vae,
        reco_loss=loss_reco,
        kld_loss=loss_kld
    )


def _setup_optimizer(config):
    optim_config = config["train"]["common"]["optimiser_config"]
    optim_name = optim_config["optmiser"]
    try:
        compute_optim = getattr(optim, optim_name)
    except AttributeError:
        print("Optimizer not found in AXO, looking in TF")
        compute_optim = getattr(tf.keras.optimizers, optim_name)

    allowed_keys = list(
        set(optim_config.keys()).intersection(set(utilities.allowed_params(compute_optim)))
    )
    config_to_optim = {k: v for k, v in optim_config.items() if k in allowed_keys}
    return compute_optim(**config_to_optim)


def _setup_callbacks(config):
    callbacks = []
    callback_config = config["callback"]

    try:
        lrsc_config = callback_config["lr_schedule"]
    except KeyError:
        print("Learning rate scheduler not found; training with constant learning rate.")
        return callbacks

    lrsc_name = lrsc_config["name"]
    lrsc_config = lrsc_config["config"]

    try:
        compute_lrsc = getattr(axo_callbacks.lr_scheduler, lrsc_name)
    except AttributeError:
        print("Scheduler not found in AXO, looking in TF callbacks")
        compute_lrsc = getattr(tf.keras.callbacks, lrsc_name)

    allowed_keys = list(
        set(lrsc_config.keys()).intersection(set(utilities.allowed_params(compute_lrsc)))
    )
    config_to_lrsc = {k: v for k, v in lrsc_config.items() if k in allowed_keys}

    def wrapped_lr_schedule(epoch, lr):
        return float(compute_lrsc(**config_to_lrsc)(epoch, lr))

    callbacks.append(tf.keras.callbacks.LearningRateScheduler(wrapped_lr_schedule, verbose=1))
    callbacks.append(BitwidthLogger())
    callbacks.append(HardwareLogger())
    return callbacks


def _trim_encoder(vae, config):
    model = Model(
        inputs=vae.encoder.input,
        outputs=vae.encoder.get_layer("latent_mean").output
    )
    print("\n[INFO] Encoder-to-mean model summary:")
    model.summary()
    model.save(config["trimmed_encoder"])
    print("\n[INFO] Model saved")
    return model


def _generate_results(config, vae, model, history):
    config_thres = config["threshold"]
    config_thres["data_path"] = config["data_config"]["Processed_data_path"]

    axo_man = metric.axo_threshold_manager(
        model=model,
        config=config_thres,
        additional_metrics=config["additional_metrics"]
    )
    dist_plot = metric.distribution_plots(model=model, config=config_thres)

    utilities.store_axo(
        config=config,
        model=vae,
        model_trim=model,
        axo_man=axo_man,
        dist_plot=dist_plot,
        history_dict=history.history
    )

    threshold_dict = utilities.retrieve.get_threshold_dict(config["store"]["lite_path"])
    dict_axo = utilities.retrieve.get_axo_score_dataframes(config["store"]["lite_path"])
    histogram_dict = utilities.retrieve.get_histogram_dict(config["store"]["lite_path"])
    history_dict = utilities.retrieve.get_history_dict(config["store"]["lite_path"])

    report_config = config["report"]
    if report_config["html_report"]["generate"] 
        print("Report generation flag found !!")
        utilities.generate_axolotl_html_report(
            config=config,
            dict_axo=dict_axo,
            histogram_dict=histogram_dict,
            threshold_dict=threshold_dict,
            history_dict=history_dict,
            output_file=report_config["html_report"]["path"],
        )
