from hgq.utils.sugar.pareto import ParetoFront as _ParetoFront
from keras.models import Model


class EncoderParetoFront(_ParetoFront):
    """ParetoFront that saves only the trimmed encoder (latent_mean output)."""

    def on_epoch_end(self, epoch, logs=None):
        # Temporarily swap self.model to the trimmed encoder for saving
        full_model = self._model
        try:
            trimmed = Model(
                inputs=full_model.encoder.input,
                outputs=full_model.encoder.get_layer("latent_mean").output
            )
            self._model = trimmed
            super().on_epoch_end(epoch, logs)
        finally:
            self._model = full_model
