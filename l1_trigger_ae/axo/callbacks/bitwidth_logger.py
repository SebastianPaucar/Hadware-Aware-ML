import tensorflow as tf
import numpy as np
from tensorflow import keras

class BitwidthLogger(keras.callbacks.Callback):
    """Callback para registrar bitwidths en el log de entrenamiento."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        bw_values = {}

        for layer in self.model.layers:
            for var in layer.trainable_variables:
                if "bw" in var.name:
                    val = float(tf.reduce_mean(var).numpy())
                    key = var.name.replace("/", "_").replace(":0", "")
                    logs[key] = val

        # promedio general de bitwidths
        if bw_values:
            logs["avg_bitwidth"] = np.mean(list(bw_values.values()))

