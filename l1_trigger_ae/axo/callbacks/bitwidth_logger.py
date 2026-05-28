import tensorflow as tf
import numpy as np
from tensorflow import keras


class BitwidthLogger(keras.callbacks.Callback):
    """
    Logs HGQ2 quantizer bitwidth statistics per epoch.

    HGQ2 uses per-tensor learnable fixed-point quantizers with
    variables named 'b' (total bits), 'i' (integer bits), 'f'
    (fractional bits). This replaces the QKeras/HGQ1 'bw' naming.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        b_values = []  # total bitwidths
        i_values = []  # integer bits
        f_values = []  # fractional bits

        for layer in self.model.layers:
            for var in layer.trainable_variables:
                name = var.name

                # HGQ2 quantizer variables are named 'b', 'i', 'f'
                # Use endswith to avoid matching e.g. 'bias'
                base = name.split("/")[-1].split(":")[0]

                val = tf.reduce_mean(var).numpy()

                if base == "b":
                    b_values.append(val)
                    key = name.replace("/", "_").replace(":0", "")
                    logs[f"bw_b_{key}"] = float(val)

                elif base == "i":
                    i_values.append(val)
                    key = name.replace("/", "_").replace(":0", "")
                    logs[f"bw_i_{key}"] = float(val)

                elif base == "f":
                    f_values.append(val)
                    key = name.replace("/", "_").replace(":0", "")
                    logs[f"bw_f_{key}"] = float(val)

        # Summary statistics — these go into history.history
        if b_values:
            logs["avg_total_bits"]     = float(np.mean(b_values))
            logs["min_total_bits"]     = float(np.min(b_values))
            logs["max_total_bits"]     = float(np.max(b_values))

        if i_values:
            logs["avg_integer_bits"]   = float(np.mean(i_values))

        if f_values:
            logs["avg_fractional_bits"] = float(np.mean(f_values))

        # Effective bitwidth = avg_integer_bits + avg_fractional_bits + 1 (sign)
        if i_values and f_values:
            logs["avg_effective_bits"] = float(
                np.mean(i_values) + np.mean(f_values) + 1
            )
