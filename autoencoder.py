import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential

tf.random.set_seed(123)
tf.keras.backend.set_floatx('float32')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train / 255.0


encoder = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, 3, padding='same'), 
    BatchNormalization(),
    LeakyReLU(),

    Conv2D(64, 3, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2D(64, 3, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2D(64, 3, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Flatten(),
    Dense(16)
])

encoder.summary()

decoder = Sequential([
    Input(shape=(16)),

    Dense(7*7*64), 
    Reshape((7, 7, 64)),
    
    Conv2DTranspose(64, 3, strides=1, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(64, 3, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(64, 3, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(32, 3, strides=1, padding='same'),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
])

decoder.summary()

encoder_in = Input(shape=(28, 28, 1))
x = encoder(encoder_in)
decoder_out = decoder(x)

auto_encoder = Model(encoder_in, decoder_out)
auto_encoder.compile(
    loss='mse',
    optimizer='sgd'
)

auto_encoder.fit(x_train, x_train, epochs=1)

# fully_connected_layer = Sequential([
#     Dense(10, activation='softmax')
# ])

# encoder_in = Input(shape=(28, 28, 1))
# x = encoder(encoder_in)
# mnist_out = fully_connected_layer(x)
# mnist_model = Model(encoder_in, mnist_out)
# mnist_model.compile(
#     loss='mse',
#     optimizer='sgd'
# )

# mnist_model.fit(x_train, y_train, epochs=15)

test_data = x_train[:1]
restored_data = auto_encoder(test_data)


test_data = test_data * 255
restored_data = restored_data * 255

test_data = np.reshape(test_data[0], (28, 28))
restored_data = np.reshape(restored_data[0], (28, 28))

from PIL import Image

img = Image.fromarray(test_data)
img.show()

img = Image.fromarray(restored_data)
img.show()