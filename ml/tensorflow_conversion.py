"""Optional orthodox TensorFlow -> TFLite conversion of the SAME trained weights.

With TensorFlow and NumPy installed on your PC: python ml/tensorflow_conversion.py
Output is a comparison file; the app uses the bundled spamshield.tflite.
Do not replace the bundled model without validating shapes and outputs.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
w=np.load(ROOT/'ml/weights.npz')
net=tf.keras.Sequential([
    tf.keras.Input(shape=(w['w0'].shape[1],),dtype=tf.float32),
    tf.keras.layers.Dense(w['w0'].shape[0],activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax'),
])
net.layers[0].set_weights([w['w0'].T,w['b0']])
net.layers[1].set_weights([w['w1'].T,w['b1']])
converter=tf.lite.TFLiteConverter.from_keras_model(net)
compiled=converter.convert()
out=ROOT/'ml/spamshield_converted_with_tensorflow.tflite'
out.write_bytes(compiled)
print('Saved official TensorFlow conversion:',out,'bytes:',len(compiled))
interpreter=tf.lite.Interpreter(model_content=compiled)
interpreter.allocate_tensors()
print('Input:',interpreter.get_input_details()[0]['shape'])
print('Output:',interpreter.get_output_details()[0]['shape'])
