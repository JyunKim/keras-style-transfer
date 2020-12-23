import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=100000).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

img = train_images[0]

print(img.shape)

img = img.reshape(-1, 28, 28, 1)

conv2d = tf.keras.layers.Conv2D(filters=5, kernel_size=3, strides=2, padding='same')(img) # stride는 padding same 적용하고나서 적용됨
print(conv2d.shape)

pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(conv2d)
print(pool.shape)