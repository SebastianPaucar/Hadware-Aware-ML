import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten
from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_relu

# Modelo simple con cuantización
model = Sequential([
    Input(shape=(10,)),
    QDense(16, kernel_quantizer=quantized_relu(bits=4, use_ste=True),
               bias_quantizer=quantized_relu(bits=4, use_ste=True)),
    QActivation('quantized_relu(4, use_ste=True)'),
    QDense(1, kernel_quantizer=quantized_relu(bits=4, use_ste=True),
              bias_quantizer=quantized_relu(bits=4, use_ste=True))
])

# Resumen del modelo
print("Resumen del modelo:")
model.summary()

# Dummy input/output
x = tf.random.normal((1, 10))
y_true = tf.random.normal((1, 1))

# Forward pass y gradientes
with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = tf.reduce_mean((y_pred - y_true) ** 2)

grads = tape.gradient(loss, model.trainable_weights)

# Verificación de gradientes
print("\nVerificación de gradientes:")
for w, g in zip(model.trainable_weights, grads):
    if g is None:
        print(f"No gradiente para: {w.name}")
    else:
        print(f"Gradiente OK para: {w.name}")

