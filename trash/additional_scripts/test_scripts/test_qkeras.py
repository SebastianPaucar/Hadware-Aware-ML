import tensorflow as tf
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
from tensorflow.keras import Input, Model

# Aseguramos que el modo eager esté habilitado
tf.compat.v1.enable_eager_execution()

# Creamos un modelo simple con QDense
input_layer = Input(shape=(10,))
x = QDense(8,
           kernel_quantizer=quantized_bits(8, 0, use_ste=True),
           bias_quantizer=quantized_bits(8, 0, use_ste=True),
           activation=None)(input_layer)
x = QActivation(activation=quantized_relu(8, 0, use_ste=True))(x)
model = Model(inputs=input_layer, outputs=x)

# Datos de entrada aleatorios para la prueba
x_data = tf.random.normal((4, 10))
y_target = tf.random.normal((4, 8))

# Compilamos el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Usamos GradientTape para calcular los gradientes
with tf.GradientTape() as tape:
    tape.watch(model.input)  # Aseguramos que estamos vigilando las entradas
    y_pred = model(x_data)
    loss = tf.reduce_mean(tf.square(y_pred - y_target))

# Calculamos los gradientes de las variables entrenables
grads = tape.gradient(loss, model.trainable_variables)

# Imprimimos los gradientes de cada peso
for var, grad in zip(model.trainable_variables, grads):
    print(f"Layer: {var.name} | Grad is None: {grad is None} | Grad: {grad.numpy() if grad is not None else 'No Gradient'}")

