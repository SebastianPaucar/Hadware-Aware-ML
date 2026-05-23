from qkeras import QDense, quantized_bits, QActivation, quantized_relu
from tensorflow.keras import Input, Model
import tensorflow as tf

input_layer = Input(shape=(10,))
x = QDense(8,
           kernel_quantizer=quantized_bits(8, 0, use_ste=True),
           bias_quantizer=quantized_bits(8, 0, use_ste=True),
           activation=None)(input_layer)
x = QActivation(activation=quantized_relu(8, 0, use_ste=True))(x)
model = Model(inputs=input_layer, outputs=x)

x_data = tf.random.normal((4, 10))
y_target = tf.random.normal((4, 8))

with tf.GradientTape() as tape:
    pred = model(x_data)
    loss = tf.reduce_mean(tf.square(pred - y_target))

grads = tape.gradient(loss, model.trainable_weights)
for v, g in zip(model.trainable_weights, grads):
    print(f"{v.name:40} | Grad is None: {g is None}")

