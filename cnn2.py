import tensorflow as tf
import numpy as np

tf.random.set_seed(123)

learning_rate = 0.01
training_epochs = 5
# batch_size = 100

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

train_images = np.expand_dims(train_images, axis=-1) # 4차원으로 만들어줌(mnist는 흑백이라 3차원)
test_images = np.expand_dims(test_images, axis=-1)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10) # one hot encoding
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=training_epochs, validation_data=(test_images, test_labels))

model.summary()

predict = model.predict(test_images)
print("predict: ", np.argmax(predict, axis=1))
print("answer: ", np.argmax(test_labels, axis=1))

evaluation = model.evaluate(test_images, test_labels)
print("loss: ", evaluation[0])
print("accuracy: ", evaluation[1])
