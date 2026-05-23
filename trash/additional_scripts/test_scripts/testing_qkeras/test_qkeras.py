import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits
import numpy as np

tf.config.run_functions_eagerly(True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

x_train = np.random.rand(100, 4).astype(np.float32)
y_train = np.random.rand(100, 1).astype(np.float32)

# Usa instancias explícitas con use_ste=True
kernel_q = quantized_bits(4, 0, 1, use_ste=True)
bias_q = quantized_bits(4, 0, 1, use_ste=True)

model = Sequential([
    Input(shape=(4,)),
    QDense(8,
           kernel_quantizer=kernel_q,
           bias_quantizer=bias_q,
           name="qdense"),
    QActivation("quantized_relu(4,0)", name="qact"),
    QDense(1, name="output")
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Verificación de gradientes
x = tf.convert_to_tensor(x_train[:10])
y = tf.convert_to_tensor(y_train[:10])
with tf.GradientTape() as tape:
    preds = model(x)
    loss = tf.reduce_mean(tf.square(preds - y))
grads = tape.gradient(loss, model.trainable_variables)

# Mostrar si hay gradiente
for var, grad in zip(model.trainable_variables, grads):
    print(f"{var.name}: {'Gradiente OK' if grad is not None else 'NO gradiente!'}")

# Entrenamiento
model.fit(x_train, y_train, epochs=5, batch_size=10)

