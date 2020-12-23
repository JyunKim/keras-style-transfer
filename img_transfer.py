import tensorflow as tf
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator

train = glob("DeepLearning/train/*.jpg")
validation = glob("DeepLearning/validation/*.jpg")

train_images = []
valid_images = []

for img in train:
    img = load_img(img, target_size=(128, 128))
    img = img_to_array(img).astype(np.float32)/255.
    train_images.append(img)

train_images = np.array(train_images)

for img in validation:
    img = load_img(img, target_size=(128, 128))
    img = img_to_array(img).astype(np.float32)/255.
    valid_images.append(img)

valid_images = np.array(valid_images)
