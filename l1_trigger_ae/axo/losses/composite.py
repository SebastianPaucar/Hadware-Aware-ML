import tensorflow as tf
from tensorflow.keras.losses import Loss


class CompositeLoss(Loss):
    """
    Wraps an ordered list of (loss_fn, weight) pairs and returns their
    weighted sum. Each component loss must accept (y_true, y_pred).
    """

    def __init__(self, components: list[tuple], name="composite_loss"):
        """
        Args:
            components: list of (loss_callable, weight: float) tuples
        """
        super().__init__(name=name)
        self.components = components  # [(loss_fn, weight), ...]

    def call(self, y_true, y_pred):
        total = None
        for loss_fn, weight in self.components:
            term = tf.cast(loss_fn(y_true, y_pred), dtype=tf.float32)
            weighted = float(weight) * term
            total = weighted if total is None else total + weighted
        return total

    def get_config(self):
        # weights only — the loss objects themselves aren't trivially serialisable
        return {
            "components": [(fn.name, w) for fn, w in self.components],
            "name": self.name,
        }
