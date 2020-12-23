import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, save_img
from tensorflow.keras.applications import vgg19

base_image_path = 'DeepLearning/train/IMG_4992.jpg' # 변환하려는 이미지 경로
style_reference_image_path = 'DeepLearning/train/0002.jpg' # 스타일 이미지 경로

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8
# 손실 항목의 가중치 평균에 사용할 가중치

width, height = load_img(base_image_path).size
img_height = 400
img_width = int(width * img_height / height) # 생성된 사진의 차원

def preprocess_image(image_path):
    img = load_img(image_path, target_size = (img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68 
    # imageNet의 평균 픽셀 값을 더한다. vgg19,preprocess_input 함수에서 일어나는 변환을 복원하는것
    x = x[:, :, ::-1] # 이미지를 'BGR'에서 'RGB'로 변환. 이것도 vgg19,preprocess_input 함수에서 일어나는 변환을 복원하는것
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4. * (channels**2) * (size**2))

def total_variation_loss(x):
    a = tf.square(
        x[:, :img_height -1 , :img_width -1, :] -
        x[:, 1:, :img_width-1, :])
    b = tf.square(
        x[:, :img_height -1 , :img_width -1, :] -
        x[:, :img_height-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)

content_layer = "block5_conv2" #콘텐츠 손실에 사용할 층
style_layers = ["block1_conv1",
              "block2_conv1",
              "block3_conv1",
              "block4_conv1",
              "block5_conv1"]

def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layers:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

optimizer = tf.keras.optimizers.SGD(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 2000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))

img = deprocess_image(combination_image.numpy())
fname = "combination_img.jpg"
save_img(fname, img)