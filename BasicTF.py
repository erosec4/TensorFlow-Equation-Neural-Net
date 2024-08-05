import tensorflow as tf
import keras
from keras import layers
import os

# Turn off environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PYTHONUTF8=1 # ???


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Pixels range 0-255, so scale to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0 


# Build a sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Return a vector of logits/log-odds scores (1 score for each class)
predictions = model(x_train[:1]).numpy()
print('\nPREDICTIONS')
predictions

# Convert logits to probabilities (normalization)
print('\nSOFTMAX')
tf.nn.softmax(predictions).numpy()

# Define a loss function for training
''' = negative log probability of the true class 
Here: probability ground truth is logits = 1/10
Initial loss = -log(1/10) ~= 2.3'''
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print('\nLOSS')
loss_fn(y_train[:1], predictions).numpy()
print('\n')

# Configure and compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train and evaluate the model
# fit method adjusts the model parameters to minimize loss
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)


'''2 WAYS TO DO LAYERS FOR A SEQUENTIAL MODEL: 

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)


# OR


# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))

'''