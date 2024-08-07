import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

# Turn off environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize the sequential model (1 input -> 1 output tensor)
model = keras.Sequential([
        keras.Input(shape=[1]), 
        keras.layers.Dense(1) # 1 unit (1 dimensional)
    ])

# Loss measures guess against actual answer
# Optimizer makes another guess to minimize the loss (sgd = stochastic gradient descent)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Use NumPy to provide data
# Correct relationship: y = 3x + 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the NN
# Fit method "fits xs to the ys 500 times"
model.fit(xs, ys, epochs=500)

# Predict
print(model.predict(float(10.0)))