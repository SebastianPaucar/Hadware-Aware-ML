import tensorflow as tf

class PruningMetrics(tf.keras.callbacks.Callback):
    """
    Callback que registra métricas relacionadas con poda por bitwidth.
    """
    def __init__(self, prune_threshold=2.0):
        super().__init__()
        self.prune_threshold = prune_threshold
        self.history = {}  # Diccionario para almacenar métricas por época

    def on_train_begin(self, logs=None):
        self.history = {
            "pruned_ratio": [],
            "effective_params": [],
            "pruned_params": []
        }

    def on_epoch_end(self, epoch, logs=None):
        total_params = 0
        pruned_params = 0

        for layer in self.model.layers:
            for var in layer.trainable_variables:
                if "bw" in var.name:  # Solo variables de bitwidth
                    bw = tf.convert_to_tensor(var).numpy()
                    total_params += bw.size
                    pruned_params += (bw < self.prune_threshold).sum()

        pruned_ratio = pruned_params / total_params if total_params > 0 else 0
        effective_params = total_params - pruned_params

        # Guardar en logs para que se pueda acceder desde history
        if logs is not None:
            logs["pruned_ratio"] = pruned_ratio
            logs["effective_params"] = effective_params
            logs["pruned_params"] = pruned_params

        # Guardar en el diccionario del callback
        self.history["pruned_ratio"].append(pruned_ratio)
        self.history["effective_params"].append(effective_params)
        self.history["pruned_params"].append(pruned_params)

        print(f"\nEpoch {epoch+1} pruning metrics: pruned_ratio={pruned_ratio:.4f}, "
              f"effective_params={effective_params}, pruned_params={pruned_params}")

