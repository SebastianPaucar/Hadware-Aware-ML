import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qkeras.quantizers import quantized_bits, quantized_relu

# Rango de entrada más fino
x = tf.linspace(-2.0, 2.0, 400)

# Cuantizadores
qb = quantized_bits(bits=4, integer=0, use_ste=True)
qr = quantized_relu(bits=4, integer=0, use_ste=True)
qrl = quantized_relu(bits=4, integer=0, use_ste=True, linear=True)

# Cuantización
x_qb = qb(x)
x_qr = qr(x)
x_qrl = qrl(x)

# Convertir a numpy para graficar
x_np = x.numpy()
x_qb_np = x_qb.numpy()
x_qr_np = x_qr.numpy()
x_qrl_np = x_qrl.numpy()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_np, x_np, 'k--', label='Identidad (x)')
plt.plot(x_np, x_qb_np, label='quantized_bits')
plt.plot(x_np, x_qr_np, label='quantized_relu')
plt.plot(x_np, x_qrl_np, label='quantized_relu (linear=True)')
plt.title('Comparación de cuantizadores QKeras (bits=4, integer=0)')
plt.xlabel('Entrada')
plt.ylabel('Salida cuantizada')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar la figura
plt.savefig("quantization_comparison.png", dpi=300)
plt.show()
