import numpy as np
import tensorflow as tf
K = tf.keras.backend
from .base import L1ADBaseLoss


class _cyl_PtPzPhi_mae_loss(L1ADBaseLoss):
    """
    Reconstruction loss over pT, pz, and phi.

    - pT:  direct MSE between true and predicted transverse momentum
    - pz:  MSE between true and predicted longitudinal momentum,
           computed as pz = pT * sinh(eta) using PREDICTED eta (fixed
           relative to the legacy cyl_PtPz which used true eta)
    - phi: direct MSE between true and predicted azimuthal angle

    """

    def __init__(self, norm_scales, norm_biases, mask,
                 unscale_energy=False, name="CylPtPzPhi"):
        super().__init__(norm_scales, norm_biases, mask,
                         unscale_energy, name=name)

    def call(self, y_true, y_pred):

        y_pred = K.reshape(y_pred, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.reshape(y_true, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.cast(y_true, dtype='float32') * self.scales + self.biases
        y_pred = K.cast(y_pred, dtype='float32') * self.scales + self.biases

        pt,      eta,      phi      = y_true[:, :, 0], y_true[:, :, 1], y_true[:, :, 2]
        pt_pred, eta_pred, phi_pred = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2]

        pz      = pt      * tf.math.sinh(eta)
        pz_pred = pt_pred * tf.math.sinh(eta_pred)  # fixed: uses predicted eta

        return K.mean(
            K.square(pt - pt_pred)
            + K.square(pz - pz_pred)
            + K.square(phi - phi_pred),
            axis=1
        )
