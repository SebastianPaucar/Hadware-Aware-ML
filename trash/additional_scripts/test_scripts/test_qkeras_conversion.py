import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from qkeras import QDense, QActivation, quantized_bits, quantized_relu

# Datos dummy
x_data = tf.random.normal((100, 10))
y_data = tf.random.normal((100, 1))

# ----------------------------
# 1. Modelo sin cuantización
# ----------------------------
inputs = Input(shape=(10,))
x = Dense(16, activation='relu')(inputs)
outputs = Dense(1)(x)
model_fp = Model(inputs, outputs)

model_fp.compile(optimizer='adam', loss='mse')
model_fp.fit(x_data, y_data, epochs=5, verbose=0)

# ----------------------------
# 2. Modelo cuantizado con QKeras
# ----------------------------
inputs_q = Input(shape=(10,))
x_q = QDense(16,
             kernel_quantizer=quantized_bits(8, 0, use_ste=True),
             bias_quantizer=quantized_bits(8, 0, use_ste=True))(inputs_q)
x_q = QActivation(quantized_relu(8, 0, use_ste=True))(x_q)
outputs_q = QDense(1,
                   kernel_quantizer=quantized_bits(8, 0, use_ste=True),
                   bias_quantizer=quantized_bits(8, 0, use_ste=True))(x_q)
model_q = Model(inputs_q, outputs_q)

# Copiar pesos del modelo float al cuantizado
model_q.set_weights(model_fp.get_weights())

# ----------------------------
# 3. Comparar outputs
# ----------------------------
x_test = tf.random.normal((5, 10))
y_fp = model_fp(x_test)
y_q = model_q(x_test)

print("Output modelo float:")
print(y_fp.numpy())
print("\nOutput modelo cuantizado:")
print(y_q.numpy())

