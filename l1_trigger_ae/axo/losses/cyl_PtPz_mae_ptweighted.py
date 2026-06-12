import numpy as np
import tensorflow as tf
K = tf.keras.backend
from .base import L1ADBaseLoss

class _cyl_PtPz_mae_ptweighted_loss(L1ADBaseLoss):

    def __init__(self, norm_scales, norm_biases, mask, unscale_energy=False,
                 gamma=0.1, max_weight=5.0, name="Cyl_PtPz_mae_ptweighted"):
        super().__init__(norm_scales, norm_biases, mask, unscale_energy, name=name)
        self.gamma      = float(gamma)
        self.max_weight = float(max_weight)

    def call(self, y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.reshape(y_true, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.cast(y_true, dtype='float32') * self.scales + self.biases
        y_pred = K.cast(y_pred, dtype='float32') * self.scales + self.biases

        pt,  eta = y_true[:, :, 0], y_true[:, :, 1]
        pz = pt * tf.math.sinh(eta)

        pt_pred = y_pred[:, :, 0]
        pz_pred = pt_pred * tf.math.sinh(eta)   # legacy convention, kept as-is

        per_constituent_error = K.abs(pt - pt_pred) + K.abs(pz - pz_pred)  # (batch, N)

        # ── pT-based weight, per constituent ──────────────────────────
        pt_safe = tf.maximum(pt, 0.0)                       # guard against negatives
        weight  = 1.0 + self.gamma * tf.math.log1p(pt_safe) # log scale — tames heavy tail
        weight  = tf.clip_by_value(weight, 1.0, self.max_weight)

        weighted_error = weight * per_constituent_error     # (batch, N)

        return K.mean(weighted_error, axis=1)
