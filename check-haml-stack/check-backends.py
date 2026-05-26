import tensorflow as tf
import torch
import jax

print('TensorFlow:', tf.__version__,
      '| GPU:', tf.config.list_physical_devices('GPU'))
print('PyTorch:', torch.__version__,
      '| cuDNN:', torch.backends.cudnn.version(),
      '| GPU:', torch.cuda.is_available())
print('JAX:', jax.__version__,
      '| Devices:', jax.devices(),
      '| Backend:', jax.default_backend())
