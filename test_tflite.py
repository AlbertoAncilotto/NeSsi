import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import nessi

model = models.Sequential()
model.add(layers.Conv2D(46, 1, activation='relu', use_bias=False, input_shape=(320, 320, 3)))
model.add(layers.BatchNormalization())
model.add(layers.DepthwiseConv2D(3, activation='relu', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(16, 1, use_bias=False))
model.add(layers.Conv2D(46, 1, activation='relu', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.DepthwiseConv2D(3, activation='relu', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(12, 1, use_bias=False))

model.summary()


def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 320, 320, 3)
      yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Get model size from bytearray data
nessi.get_model_size(tflite_model, 'tflite')

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# Get model size from model path
nessi.get_model_size('model.tflite', 'tflite')


