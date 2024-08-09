# Using Fashion MNIST Dataset (70k images including test images, 28x28 pixels each)
# 10 categories, labeled by numbers

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Takes in 28 x 28 pixels, output 1 of 10 category values
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 1st layer: size of image
    keras.layers.Dense(128, activation='relu'), # 2nd layer: ?
    keras.layers.Dense(10, activation='softmax') # 3rd layer: number of categories
])

'''
Relu: rectified __ unit;
Softmax: picks the biggest number in a set (probable category in array = 1, rest are 0)
'''

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss)
print(test_acc)

# predictions = model.predict(my_images)