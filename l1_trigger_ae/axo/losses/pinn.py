import numpy as np
import tensorflow as tf
K = tf.keras.backend
from .base import L1ADBaseLoss


class _pinn_loss(L1ADBaseLoss):
    """
    Physics-Informed reconstruction loss.

    Combines the cyl_PtPz base loss with a momentum conservation
    penalty. The penalty enforces that the reconstructed event preserves
    the total transverse momentum vector (sum of px, py) of the input.

    """

    def __init__(self, norm_scales, norm_biases, mask,
                 unscale_energy=False, name="PINN_loss"):
        super().__init__(norm_scales, norm_biases, mask,
                         unscale_energy, name=name)

    def call(self, y_true, y_pred):
        # ── Unscale exactly as cyl_PtPz_mae does ─────────────────────
        y_pred = K.reshape(y_pred, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.reshape(y_true, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.cast(y_true, dtype='float32') * self.scales + self.biases
        y_pred = K.cast(y_pred, dtype='float32') * self.scales + self.biases

        # ── Kinematic quantities ──────────────────────────────────────
        pt,      eta  = y_true[:, :, 0], y_true[:, :, 1]
        phi_true      = y_true[:, :, 2]
        pt_pred       = y_pred[:, :, 0]
        phi_pred      = y_pred[:, :, 2]

        # ── Momentum conservation penalty ─────────────────────────────
        # px = pT * cos(phi),  py = pT * sin(phi)
        # Penalty: the reconstructed event must preserve the total
        # transverse momentum vector of the input event.
        px_true = pt      * tf.cos(phi_true)
        py_true = pt      * tf.sin(phi_true)
        px_pred = pt_pred * tf.cos(phi_pred)
        py_pred = pt_pred * tf.sin(phi_pred)

        # Sum over constituents total event momentum
        total_px_true = tf.reduce_sum(px_true, axis=1)   # (batch,)
        total_py_true = tf.reduce_sum(py_true, axis=1)
        total_px_pred = tf.reduce_sum(px_pred, axis=1)
        total_py_pred = tf.reduce_sum(py_pred, axis=1)

        momentum_penalty = (
            tf.square(total_px_true - total_px_pred)
            + tf.square(total_py_true - total_py_pred)
        )   # (batch,)

        return momentum_penalty / tf.cast(self.NOF_CONSTITUENTS, tf.float32)
