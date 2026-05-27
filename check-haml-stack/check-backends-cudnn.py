import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras  # keras FIRST
import tensorflow as tf
import torch
import jax

print('Keras:      ', keras.__version__)
print('TensorFlow: ', tf.__version__)
print('PyTorch:    ', torch.__version__)
print('JAX:        ', jax.__version__)

import ctypes
lib = ctypes.cdll.LoadLibrary('libcudnn.so.9')
cudnn_ver = lib.cudnnGetVersion()
print(f'cuDNN lib:   {cudnn_ver // 1000}.{(cudnn_ver % 1000) // 100}.{cudnn_ver % 100}')

import importlib.metadata
for pkg in ['nvidia-cudnn-cu12', 'jaxlib', 'jax-cuda12-plugin', 'keras']:
    try:
        print(f'{pkg}: {importlib.metadata.version(pkg)}')
    except:
        print(f'{pkg}: not installed')
