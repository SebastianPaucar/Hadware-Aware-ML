import keras
import numpy as np


def _iter_quantized_layers(model):
    """Recursively yield (layer_name_path, layer) for all layers with HGQ2 quantizer vars."""
    for layer in model.layers:
        # Recurse into sub-models
        if hasattr(layer, 'layers'):
            yield from _iter_quantized_layers(layer)
        else:
            b_i_f = [
                v for v in layer.trainable_variables
                if v.name.split("/")[-1].split(":")[0] in ("b", "i", "f")
            ]
            if b_i_f:
                yield layer


class BitwidthLogger(keras.callbacks.Callback):
    """
    Logs HGQ2 quantizer bitwidth statistics per epoch, per quantized layer.
    Architecture-agnostic: works with any HGQ2 quantized layer (QDense, QConv, etc.)
    Recurses into sub-models (encoder, decoder) to find individual layers.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        all_b, all_i, all_f = [], [], []

        for layer in _iter_quantized_layers(self.model):
            b_vals, i_vals, f_vals = [], [], []

            for var in layer.trainable_variables:
                base = var.name.split("/")[-1].split(":")[0]
                val = float(np.mean(var.numpy()))
                if base == "b":
                    b_vals.append(val)
                elif base == "i":
                    i_vals.append(val)
                elif base == "f":
                    f_vals.append(val)

            layer_name = layer.name
            if b_vals:
                logs[f"{layer_name}/avg_total_bits"]      = float(np.mean(b_vals))
                logs[f"{layer_name}/min_total_bits"]      = float(np.min(b_vals))
                logs[f"{layer_name}/max_total_bits"]      = float(np.max(b_vals))
                all_b.extend(b_vals)
            if i_vals:
                logs[f"{layer_name}/avg_integer_bits"]    = float(np.mean(i_vals))
                all_i.extend(i_vals)
            if f_vals:
                logs[f"{layer_name}/avg_fractional_bits"] = float(np.mean(f_vals))
                all_f.extend(f_vals)
            if i_vals and f_vals:
                logs[f"{layer_name}/avg_effective_bits"]  = float(
                    np.mean(i_vals) + np.mean(f_vals) + 1
                )

        # Global summary across all quantized layers
        if all_b:
            logs["avg_total_bits"]      = float(np.mean(all_b))
            logs["min_total_bits"]      = float(np.min(all_b))
            logs["max_total_bits"]      = float(np.max(all_b))
        if all_i:
            logs["avg_integer_bits"]    = float(np.mean(all_i))
        if all_f:
            logs["avg_fractional_bits"] = float(np.mean(all_f))
        if all_i and all_f:
            logs["avg_effective_bits"]  = float(np.mean(all_i) + np.mean(all_f) + 1)
