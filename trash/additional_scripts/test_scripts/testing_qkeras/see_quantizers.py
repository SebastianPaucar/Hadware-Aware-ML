import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qkeras.quantizers import quantized_bits, quantized_relu

# Valores de prueba
x = tf.linspace(-1.5, 1.5, 10)

# Cuantizador quantized_bits con 4 bits, integer=0
qb = quantized_bits(4, 0, alpha=1,use_ste=True)
x_qb = qb(x)

# Cuantizador quantized_relu con 4 bits, integer=0 (comportamiento ReLU)
qr = quantized_relu(bits=4, integer=0, use_ste=True)
x_qr = qr(x)

qrl = quantized_relu(bits=4, integer=0, use_ste=True, linear=True)
x_qrl = qrl(x)

print(f"x_qb = {x_qb}")

print(f"x_qr = {x_qr}")

print(f"x_qrl = {x_qrl}")
