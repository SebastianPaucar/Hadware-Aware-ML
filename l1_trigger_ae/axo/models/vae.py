import numpy as np
import tensorflow as tf
import keras
from keras.layers import Layer
from keras import layers
from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.utils.sugar import FreeEBOPs
from keras.models import Model


############################################################################################
# Sampling Layer
############################################################################################

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean = tf.cast(z_mean, dtype='float32')
        z_log_var = tf.cast(z_log_var, dtype='float32')
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        epsilon = tf.cast(epsilon, dtype='float32')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


############################################################################################
# Encoder
############################################################################################

def get_encoder(config):
    encoder_config = config["encoder_config"]
    latent_dim = config["latent_dim"]
    features = config["features"]

    ap_initial_kernel = config["ap_initial_kernel"]
    ap_initial_activation = config["ap_initial_activation"]
    beta = config["ebops_beta"]   

    # Hidden layer quantizer configs
    kq = QuantizerConfig(
        q_type='kbi',
        place='weight',
        overflow_mode='SAT_SYM',
        round_mode='RND',
        b0=max(ap_initial_kernel, 8),
        i0=2
    )
    bq = QuantizerConfig(
        q_type='kbi',
        place='weight',
        overflow_mode='SAT_SYM',
        round_mode='RND',
        b0=max(ap_initial_kernel, 8),
        i0=2
    )
    iq = QuantizerConfig(
        q_type='kif',
        place='datalane',
        overflow_mode='WRAP',
        round_mode='RND',
        i0=2,
        f0=max(ap_initial_activation, 8)
    )

    # Latent layer quantizer configs — higher precision
    kq_latent = QuantizerConfig(
        q_type='kbi',
        place='weight',
        overflow_mode='SAT_SYM',
        round_mode='RND',
        b0=max(ap_initial_kernel, 12),
        i0=2
    )
    bq_latent = QuantizerConfig(
        q_type='kbi',
        place='weight',
        overflow_mode='SAT_SYM',
        round_mode='RND',
        b0=max(ap_initial_kernel, 12),
        i0=2
    )
    iq_latent = QuantizerConfig(
        q_type='kif',
        place='datalane',
        overflow_mode='WRAP',
        round_mode='RND',
        i0=2,
        f0=max(ap_initial_activation, 12)
    )

    with LayerConfigScope(enable_ebops=True, beta0=beta):

        encoder_input = keras.Input(shape=(features,))
        x = encoder_input

        for i, node in enumerate(encoder_config["nodes"]):
            x = QDense(
                node,
                activation='relu',
                name=f'hd_encoder{i+1}',
                kq_conf=kq,
                bq_conf=bq,
                iq_conf=iq
            )(x)

        z_mean = QDense(
            latent_dim,
            activation='linear',
            name='latent_mean',
            kq_conf=kq_latent,
            bq_conf=bq_latent,
            iq_conf=iq_latent
        )(x)

        z_log_var = QDense(
            latent_dim,
            activation='linear',
            name='latent_log_var',
            kq_conf=kq_latent,
            bq_conf=bq_latent,
            iq_conf=iq_latent
        )(x)

    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    return encoder


############################################################################################
# Decoder (kept as float — same as original)
############################################################################################

def get_decoder(config):
    decoder_config = config["decoder_config"]
    latent_dim = config["latent_dim"]

    decoder_input = keras.Input(shape=(latent_dim,))

    for i, node in enumerate(decoder_config["nodes"]):
        if i == 0:
            x = layers.Dense(node, name=f'hd_decoder{i+1}')(decoder_input)
        else:
            x = layers.Dense(node, name=f'hd_decoder{i+1}')(x)

        if i != len(decoder_config["nodes"]) - 1:
            x = layers.BatchNormalization(name=f'BN_decoder{i+1}')(x)
            x = layers.ReLU()(x)

    decoder = keras.Model(decoder_input, x, name="decoder")
    return decoder


############################################################################################
# Variational Autoencoder
############################################################################################

class VariationalAutoEncoderHGQ2(Model):
    def __init__(self, config, reco_loss, kld_loss):
        super().__init__()

        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)

        self.alpha = config["alpha"]
        self.beta = config["kl_beta"]
        self.reco_scale = self.alpha * (1 - self.beta)
        self.kl_scale = self.beta

        self.reco_loss = reco_loss
        self.kl_loss = kld_loss

        # Zero out latent_log_var kernel and bias at init
        log_var_layer = self.encoder.get_layer("latent_log_var")
        for var in log_var_layer.weights:
            if "kernel" in var.name or "bias" in var.name:
                var.assign(tf.zeros_like(var))

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="total_val_loss")
        self.reconstruction_val_loss_tracker = keras.metrics.Mean(name="val_reco_loss")
        self.kl_val_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data_in, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = self.reco_scale * self.reco_loss(target, reconstruction)
            kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var)
            total_loss = reconstruction_loss + kl_loss
            total_loss += tf.reduce_sum(self.encoder.losses) + tf.reduce_sum(self.decoder.losses)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        data_in, target = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)

        reconstruction_loss = self.reco_scale * self.reco_loss(target, reconstruction)
        kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + kl_loss

        self.total_val_loss_tracker.update_state(total_loss)
        self.reconstruction_val_loss_tracker.update_state(reconstruction_loss)
        self.kl_val_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_val_loss_tracker.result(),
            "reco_loss": self.reconstruction_val_loss_tracker.result(),
            "kl_loss": self.kl_val_loss_tracker.result(),
        }
