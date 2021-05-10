import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tkinter import filedialog



num_classes = 10
input_shape = (224, 224, 3)

MODEL_SAVE_FOLDER_PATH = 'C:/Users/sylee/PycharmProjects/pythonProject1/dogClassifier/ResNet50_seperate_pretrained_augO/model/'


model_input = keras.Input(shape=input_shape)
model = ResNet50(weights='imagenet', input_tensor=model_input, include_top=False)
x = layers.GlobalAvgPool2D()(model.output)
x = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(model_input, x, name='resnet50_imagenet_augO')

model.load_weights(f'{MODEL_SAVE_FOLDER_PATH}checkpoint_{model.name}.h5')

dict_dogs = {0: '닥스훈트', 1: '말티즈', 2: '시바견', 3: '웰시코기', 4: '포메라이언', 5: '보스턴테리어', 6: '비숑', 7: '치와와', 8: '스피츠', 9: '푸들'}

input_img = filedialog.askopenfilename(initialdir="/", title="Select file",
                                      filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
input_img = cv2.imread(input_img, cv2.IMREAD_COLOR)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = cv2.resize(input_img, dsize=(224, 224))
input_img = input_img.astype("float32") / 255
input_img = np.expand_dims(input_img, 0)
result = model.predict(input_img)
# print(result)
result = result[0].argmax()
print(f'{dict_dogs[result]}입니다.')


# dict_dogs = {0: '닥스훈트', 1: '말티즈', 2: '시바견', 3: '웰시코기', 4: '포메라이언', 5: '보스턴테리어', 6: '비숑', 7: '치와와', 8: '스피츠', 9: '푸들'}
#
# input_img = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                       filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
# input_img = cv2.imread(input_img, cv2.IMREAD_COLOR)
# input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# input_img = cv2.resize(input_img, dsize=(224, 224))
# input_img = input_img.astype("float32") / 255   # (244, 244, 3)
# input_img = np.expand_dims(input_img, 0)
# # print(input_img.shape)  # (1, 244, 244, 3)
# result = model.predict(input_img)
# # print(predicted_result)
# # print(predicted_result.shape)   # (1, 10)
# result = result[0].argmax()
# print(f'{dict_dogs[result]}입니다.')