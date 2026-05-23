import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling2D
from qkeras import QDense, QActivation, QConv2D, quantized_bits

# Definir cuantizador explícitamente con use_ste=True
qbits = quantized_bits(4, 0, 1, use_ste=True)

# Crear modelo con varias capas QKeras
model = Sequential([
    QConv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_quantizer=qbits, bias_quantizer=qbits,
            input_shape=(16, 16, 3)),
    QActivation(qbits),
    MaxPooling2D(),
    QConv2D(32, (3, 3), padding="same",
            kernel_quantizer=qbits, bias_quantizer=qbits),
    QActivation(qbits),
    MaxPooling2D(),
    Flatten(),
    QDense(64, kernel_quantizer=qbits, bias_quantizer=qbits),
    QActivation(qbits),
    QDense(10, kernel_quantizer=qbits, bias_quantizer=qbits)
])

# Resumen del modelo
print("\nResumen del modelo:")
model.summary()

# Mostrar pesos entrenables por capa
print("\n[INFO] Capas con pesos entrenables:")
for layer in model.layers:
    print(f"{layer.name}: {[w.name for w in layer.trainable_weights]}")

# Verificación de gradientes
x = tf.random.normal((1, 16, 16, 3))
y_true = tf.random.uniform((1,), maxval=10, dtype=tf.int32)

with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)

gradients = tape.gradient(loss, model.trainable_weights)

print("\nVerificación de gradientes:")
print("-" * 40)
for weight, grad in zip(model.trainable_weights, gradients):
    if grad is None:
        print(f"No gradiente para: {weight.name}")
    else:
        print(f"Gradiente OK para: {weight.name}")
print("-" * 40)

# Inspeccionar kernel_quantizer de la primera capa
first_conv = model.layers[0]
print("\n[DEBUG] Primera capa QConv2D:")
print("Clase del cuantizador del kernel:", type(first_conv.kernel_quantizer))
if hasattr(first_conv.kernel_quantizer, "use_ste"):
    print("¿use_ste en kernel_quantizer?:", first_conv.kernel_quantizer.use_ste)
else:
    print("El cuantizador del kernel no tiene atributo 'use_ste'")

