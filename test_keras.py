from pickletools import optimize
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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

nessi.get_model_size(model, 'keras')

