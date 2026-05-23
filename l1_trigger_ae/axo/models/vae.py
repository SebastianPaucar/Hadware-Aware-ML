import numpy as np
import tensorflow as tf
import keras
from keras.layers import Layer
from keras import layers
from HGQ.layers import HQuantize, HDense, HActivation
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf
from HGQ import ResetMinMax, FreeBOPs
from keras.models import Model
K = tf.keras.backend

tf.config.optimizer.set_jit(True)


############################################################################################
#Encoder
############################################################################################

def get_encoder(config):
    encoder_config = config["encoder_config"]
    latent_dim = config["latent_dim"]
    features = config["features"]
    
    ap_initial_kernel = config["ap_initial_kernel"]
    ap_initial_activation = config["ap_initial_activation"]
    ap_initial_data = config["ap_initial_data"]
    # beta for EBOPs
    beta = config["beta"]
    # Activation quantizer config (pre-activation)
    paq_conf = get_default_paq_conf()
    paq_conf['init_bw'] = ap_initial_activation
    # Kernel quantizer config (weights)
    kq_conf = get_default_kq_conf()
    kq_conf['init_bw'] = ap_initial_kernel

    #Activation for HQuantize layer
    dataq_conf = get_default_paq_conf()
    dataq_conf['init_bw'] = ap_initial_data

        
    encoder_input = keras.Input(shape=(features,))
    x = HQuantize(beta=beta, paq_conf=paq_conf)(encoder_input)
     
    for i,node in enumerate(encoder_config["nodes"]):
            x = HDense(node,
                      name=f'hd_encoder{i+1}',
                      activation = 'relu',
                      beta = beta,
                      kq_conf = kq_conf,
                      paq_conf = paq_conf
                     )(x)

    # Building the mean and the log var layers
    z_mean = HDense(latent_dim,
                          name=f'latent_mean',
                          activation="linear",
                          beta = beta,
                          kq_conf = kq_conf,
                          paq_conf = paq_conf
                     )(x) 

    z_log_var = HDense(latent_dim,
                          name='latent_log_var',
                          activation="linear",
                          beta = beta,
                          kq_conf = kq_conf,
                          paq_conf = paq_conf
                    )(x)

    z = Sampling()([z_mean,z_log_var])

    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    return encoder

############################################################################################
# Decoder
############################################################################################


def get_decoder(config):
    
    decoder_config = config["decoder_config"]
    latent_dim = config["latent_dim"]

    decoder_input = keras.Input(shape=(latent_dim,))
    
    for i,node in enumerate(decoder_config["nodes"]):
        if i == 0:
            x = layers.Dense(node,name=f'hd_decoder{i+1}')(decoder_input)
        
        else:
            if i == len(decoder_config["nodes"])-1: ## This is done to prevent blowup
                x = layers.Dense(node,
                          name=f'hd_decoder{i+1}',
                         )(x)
            else:
                x = layers.Dense(node,
                          name=f'hd_decoder{i+1}'
                         )(x)


        if i!=len(decoder_config["nodes"])-1:
            x = layers.BatchNormalization(name=f'BN_decoder{i+1}')(x)
            x = layers.ReLU()(x)
    
    decoder = keras.Model(decoder_input,x, name="decoder")

    return decoder


############################################################################################
# SAMPLING LAYER
############################################################################################

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean = K.cast(z_mean, dtype='float32')
        z_log_var = K.cast(z_log_var, dtype='float32')
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        epsilon = K.cast(epsilon, dtype='float32')
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
############################################################################################
#Variational Auto Encoder
############################################################################################

class VariationalAutoEncoderHGQ(Model):
    def __init__(self, config, reco_loss, kld_loss):
        super().__init__()
        encoder_config = config["encoder_config"]
        decoder_config = config["decoder_config"]
        latent_dim = config["latent_dim"]
        features = config["features"]
        
        # Peparing the encoder and the decoder >>>>
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        #### Loss bussiness .....
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        
        self.reco_scale = self.alpha * (1 - self.beta)
        self.kl_scale = self.beta
        
        self.reco_loss = reco_loss
        self.kl_loss = kld_loss
        
        #### Metrics .....
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="total_val_loss")
        self.reconstruction_val_loss_tracker = keras.metrics.Mean(name="val_reco_loss")
        self.kl_val_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        ###.........

        ##### Taken from Chang
#        log_var_k, log_var_b = self.encoder.get_layer('latent_log_var').get_weights()
#        self.encoder.get_layer('latent_log_var').set_weights([log_var_k*0, log_var_b*0])
        #####

        ##### Refactoring Chang results for HGQ
        log_var_latent_layer = self.encoder.get_layer("latent_log_var").weights
        for var in log_var_latent_layer:
            # zero only the kernel and bias
            if "kernel" in var.name or "bias" in var.name:
                var.assign(tf.zeros_like(var))
        #####
    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data_in, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = self.reco_scale * self.reco_loss(target, reconstruction)  # one value
            kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var) 
            total_loss = reconstruction_loss + kl_loss
            total_loss += K.sum(self.encoder.losses) + K.sum(self.decoder.losses)
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
        # validation
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
            "kl_loss": self.kl_val_loss_tracker.result()
        }



