# setup
import numpy as np
from keras import models
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
# prediction
import os
import shutil
import cv2
from tqdm import tqdm   # progress bar
# import random
# import matplotlib
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import applications

import math
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tkinter import filedialog


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# 케라스 기본 개념
# dataset 생성 -> 모델 구성하기 -> 모델 학습과정 설정(compile) -> 모델 학습(fit) -> 학습과정 살펴보기 -> 모델 평가하기(evaluate) -> 모델 사용(predict)
# 1. dataset 생성
#   훈련, 검증, 테스트를 위한 데이터
# 2. 모델 구성
#   시퀀스 모델 생성한 다음 레이어 추가 (간단한 모델)
#   복잡한 모델은 케라스 함수 API 사용
# 3. 모델 학습과정 설정
#   cost 함수 정의, 최적화 방법 정의
#   compile 함수 사용


## dog_img
# 0: 닥스훈트, 1: 말티즈, 2: 시바견, 3: 웰시코기, 4: 포메라이언, 5: 보스턴테리어, 6: 비숑, 7: 치와와, 8: 스피츠, 9: 푸들
dict_dogs = {0: '닥스훈트', 1: '말티즈', 2: '시바견', 3: '웰시코기', 4: '포메라이언', 5: '보스턴테리어', 6: '비숑', 7: '치와와', 8: '스피츠', 9: '푸들'}
## dog_img2
# 0: 닥스훈트, 1: 시바견, 2: 웰시코기, 3: 포메라이언, 4: 보스턴테리어, 5: 치와와, 6: 푸들

num_classes = 10
# input_shape = (128, 128, 3)
input_shape = (224, 224, 3)

# path_dir = 'C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img'
# list_file_name = os.listdir(path_dir)
# np.random.seed(10)
# np.random.shuffle(list_file_name)

#
# class_list = []
#
# for file in file_list:
#     class_list.append(file.split('(')[0])
#
# print("name_list: ", file_list)   # ['0(0).jpg', '0(1).jpg', '0(10).jpg', '0(100).jpg', '0(101).jpg', ...]
# print("class_list: ", class_list) # ['0', '0', '0', '0', '0', '0', '0', '0', ...]
#
# np_name = np.asarray(file_list)
# np_class = np.asarray(class_list)
# np_class = np_class.astype(np.int32)
#
# print(type(np_name))
# print(type(np_class))
# print("np_name: ", np_name.shape)   # (8749,)
# print("np_class: ", np_class.shape)   # (8749,)


# images = []
# labels = []
#
# mov_files = []
#
# for file in tqdm(list_file_name): # [:10]
#     path = os.path.join(path_dir, file)
#     img = cv2.imread(path)
#     if img is None:
#         mov_files.append(file)
#         continue
#     # 이미지 가로 세로 비율이 2배 이상인 것
#     elif img.shape[0]/img.shape[1] >= 2 or img.shape[1]/img.shape[0] >= 2:
#         mov_files.append(file)
#         continue
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img, dsize=(128, 128))
#     img = cv2.resize(img, dsize=(224, 224))
#     images.append(img)
#     labels.append(file.split('(')[0])
#
# # 읽어오지 못하는 파일, 비율 안 맞는 파일 옮기기
# for mov_file in mov_files:
#     print(mov_file)
#     shutil.move('C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img2' + '/' + mov_file, 'C:/Users/sylee/Documents/Laon People/project_dogClassification/mov_img2' + '/' + mov_file)
#
# images = np.asarray(images)
# print("images shape: ", images.shape)       # (8453, 244, 244, 3)
#
# labels = np.asarray(labels)
# labels = labels.astype(np.int32)
#
# print("indice shape: ", labels.shape)     # (8453,)
#
#
# print("\nclass별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(labels)):
#     num_check[labels[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))
#

#
# # training set, validation set, test set 분리 (앞 80%: training set, 나머지: test set)

# training_num = int(len(labels)/5 * 4)
# validation_num = int(training_num/10)
#
#
# # training set
# training_set_img = images[:training_num-validation_num]
# training_set_label = labels[:training_num-validation_num]
# # training_set_fname = list_file_name[:training_num-validation_num]
#
# print("\ntraining set의 class별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(training_set_label)):
#     num_check[training_set_label[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))
#
#
# # validation set
# validation_set_img = images[training_num-validation_num+1:training_num]
# validation_set_label = labels[training_num-validation_num+1:training_num]
# # validation_set_fname = list_file_name[training_num-validation_num+1:training_num]
#
#
# print("\nvalidation set의 class별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(validation_set_label)):
#     num_check[validation_set_label[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))
#
#
# # test set
# test_set_img = images[training_num:]
# test_set_label = labels[training_num:]
# test_set_fname = list_file_name[training_num:]
#
# print("\ntest set의 class별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(test_set_label)):
#     num_check[test_set_label[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))



## training set, validation set 가져오기
# path_dir = 'C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img'
path_dir_train = 'C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img_2/train'
path_dir_valid = 'C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img_2/valid'

list_file_name_train = os.listdir(path_dir_train)
list_file_name_valid = os.listdir(path_dir_valid)

images_train = []
labels_train = []

mov_files_train = []

for file in tqdm(list_file_name_train): # [:10]
    path = os.path.join(path_dir_train, file)
    img = cv2.imread(path)
    if img is None:
        mov_files_train.append(file)
        continue
    # 이미지 가로 세로 비율이 2배 이상인 것
    elif img.shape[0]/img.shape[1] >= 2 or img.shape[1]/img.shape[0] >= 2:
        mov_files_train.append(file)
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224))
    images_train.append(img)
    labels_train.append(file.split('(')[0])

images_valid = []
labels_valid = []

mov_files_valid = []

for file in tqdm(list_file_name_valid): # [:10]
    path = os.path.join(path_dir_valid, file)
    img = cv2.imread(path)
    if img is None:
        mov_files_valid.append(file)
        continue
    # 이미지 가로 세로 비율이 2배 이상인 것
    elif img.shape[0]/img.shape[1] >= 2 or img.shape[1]/img.shape[0] >= 2:
        mov_files_valid.append(file)
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, dsize=(128, 128))
    img = cv2.resize(img, dsize=(224, 224))
    images_valid.append(img)
    labels_valid.append(file.split('(')[0])

images_train = np.asarray(images_train)
print("images_train shape: ", images_train.shape)       # (8453, 244, 244, 3)

labels_train = np.asarray(labels_train)
labels_train = labels_train.astype(np.int32)


images_valid = np.asarray(images_valid)
print("images_valid shape: ", images_valid.shape)       # (8453, 244, 244, 3)

labels_valid = np.asarray(labels_valid)
labels_valid = labels_valid.astype(np.int32)

for mov_file in mov_files_train:
    print(mov_file)
    shutil.move('C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img_2/train' + '/' + mov_file, 'C:/Users/sylee/Documents/Laon People/project_dogClassification/mov_img_2/train' + '/' + mov_file)

for mov_file in mov_files_valid:
    print(mov_file)
    shutil.move('C:/Users/sylee/Documents/Laon People/project_dogClassification/dog_img_2/valid' + '/' + mov_file, 'C:/Users/sylee/Documents/Laon People/project_dogClassification/mov_img_2/valid' + '/' + mov_file)

# Scale images to the [0, 1] range
# 정규화(normalization) : 값 전체의 범위가 어떤 형태의 범위 안에 항상 들어가도록 preprocessing하는 방법
images_train = images_train.astype("float32") / 255
images_valid = images_valid.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
print("training_set_img shape:", images_train.shape)
print("training_set_label shape:", labels_train.shape)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)  # (60000, 28, 28, 1) # transpose, reshape, stack, np.array(list_image).shape -> (60000, 28*28) ->
# print("y_train shape:", y_train.shape)  # (60000,)
print(images_train.shape[0], "train samples") # 데이터 개수

# convert class vectors to binary class matrices
# ont-hot encoding
labels_train = keras.utils.to_categorical(labels_train, num_classes)
labels_valid = keras.utils.to_categorical(labels_valid, num_classes)


# # Scale images to the [0, 1] range
# # 정규화(normalization) : 값 전체의 범위가 어떤 형태의 범위 안에 항상 들어가도록 preprocessing하는 방법
# training_set_img = training_set_img.astype("float32") / 255
# test_set_img = test_set_img.astype("float32") / 255
#
# # Make sure images have shape (28, 28, 1)
# print("training_set_img shape:", training_set_img.shape)
# print("training_set_label shape:", training_set_label.shape)
# # x_train = np.expand_dims(x_train, -1)
# # x_test = np.expand_dims(x_test, -1)
# # print("x_train shape:", x_train.shape)  # (60000, 28, 28, 1) # transpose, reshape, stack, np.array(list_image).shape -> (60000, 28*28) ->
# # print("y_train shape:", y_train.shape)  # (60000,)
# print(training_set_img.shape[0], "train samples") # 데이터 개수
# print(test_set_img.shape[0], "test samples")
#
# # convert class vectors to binary class matrices
# # ont-hot encoding
# training_set_label = keras.utils.to_categorical(training_set_label, num_classes)
# test_set_label = keras.utils.to_categorical(test_set_label, num_classes)
# validation_set_label = keras.utils.to_categorical(validation_set_label, num_classes)


# build the model
# 모델 구성
model_input = keras.Input(shape=input_shape)

## ResNet50 - weights='imagenet'
# 사전학습 된 모델이란, 내가 풀고자 하는 문제와 비슷하면서 사이즈가 큰 데이터로 이미 학습이 되어 있는 모델.
# 그런 큰 데이터로 모델을 학습시키는 것은 오랜 시간과 연산량이 필요하므로, 관례적으로는 이미 공개되어있는 모델들을 그저 import해서 사용.
model = ResNet50(weights='imagenet', input_tensor=model_input, include_top=False) # include_top=False => GAP전까지
# model = ResNet50(weights=None, input_tensor=model_input, include_top=False) # include_top=False => GAP전까지
x = layers.GlobalAvgPool2D()(model.output)
x = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(model_input, x, name='resnet50_none_augO')

## VGG16
# model = VGG16(weights=None, input_tensor=model_input, include_top=True)
# #x = layers.Flatten()(model.output)
# #x = layers.Dense(4096, activation="relu")(x)
# #x = layers.Dropout(0.5)(x)
# #x = layers.Dense(4096, activation="relu")(x)
# #x = layers.Dropout(0.5)(x)
#
# x = layers.Dense(num_classes, activation="softmax")(model.layers[-2].output)
#
# model = models.Model(model_input, x)


'''
model = keras.Sequential(
    [
        # keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # # layers.Flatten(),
        # layers.GlobalAvgPool2D(),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, activation="softmax"),
    ]
)
'''
# model = keras.Sequential(
#     [
#         # VGG16 직접 구성
#         keras.Input(shape=input_shape),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=2), # 학습파라미터가 없으므로 layer로 취급하지 않는다.
#         layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding=1),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#         layers.Flatten(),
#         # layers.GlobalAvgPool2D(),
#         layers.Dense(4096, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(4096, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# 4096, 4096, 10

# input -> Conv2D -> Conv2D -> MaxPooling2D -> Conv2D -> Conv2D -> MaxPooling2D -> Conv2D -> Conv2D

model.summary()

# train the model
# 모델 훈련
# batch_size = 4
batch_size = 16 # 처음에 128이었는데, GPU 용량이 작아서 OOM(out of memory) 오류가 뜸. 그래서 32로 줄임.
epochs = 100 # 훈련 횟수. 전체 데이터를 15번 사용해서 학습을 하는 것.

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= ["accuracy"]) # sgd


train_dg = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=20,
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            # brightness_range=[0.5, 1.5],
                                                            shear_range=0.2,
                                                            zoom_range=0.2,
                                                            fill_mode='constant',
                                                            cval=0.0, # fill_mode = "constant"일 때 씀
                                                            horizontal_flip=True,
                                                            # vertical_flip=True,
                                                            )

# filename_in_dir = []
#
# for root, dirs, files in os.walk('./spider'):
#     for fname in files:
#         full_fname = os.path.join(root, fname)
#         filename_in_dir.append(full_fname)
#
# for file_img in filename_in_dir:
#     print(file_img)
#     img = load_img(file_img)
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
# i = 0
# for batch in train_dg.flow(x, save)

# validation_dg = tf.keras.preprocessing.image.ImageDataGenerator() # 이미지 픽셀값을 0~1 값으로 맞춰주기 위해서 설정


'''
train_generator = train_dg.flow_from_directory( 'C:/Users/sylee/Documents/Laon People/project_dogClassification/train_set',
                                                target_size=(224, 224),
                                                batch_size=16,
                                                class_mode='categorical')

validation_generator = validation_dg.flow_from_directory( 'C:/Users/sylee/Documents/Laon People/project_dogClassification/test_set',
                                                target_size=(224, 224),
                                                batch_size=16,
                                                class_mode='categorical')
'''


MODEL_SAVE_FOLDER_PATH = './dogClassifier/ResNet50_seperate_none_augO/model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

checkpoint = ModelCheckpoint(filepath=f'{MODEL_SAVE_FOLDER_PATH}checkpoint_{model.name}.h5', monitor='val_loss', verbose=1, save_best_only=True)

# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=16, verbose=1, mode="auto")


# history = model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, reduce_lr])
# history = model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, validation_data=(images_valid, labels_valid), callbacks=[checkpoint, reduce_lr, early_stopping])

# history = model.fit_generator(train_dg.flow(images_train, labels_train, batch_size=batch_size),
#                               steps_per_epoch=math.ceil(labels_train.shape[0]/batch_size),
#                               epochs=epochs,
#                               validation_data=(images_valid, labels_valid),
#                               callbacks=[checkpoint, reduce_lr, early_stopping])

model.load_weights(f'{MODEL_SAVE_FOLDER_PATH}checkpoint_{model.name}.h5')


## evaluate
# 모델 평가: 준비된 test set으로 학습한 모델을 평가한다.
print("test_set_img1 shape: ", images_valid.shape) # (1452, 224, 224, 3)
print("test_set_label1 shape: ", labels_valid.shape) # (1452, 10)
score = model.evaluate(images_valid, labels_valid, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

## predict
predicted_result = model.predict(images_valid)
predicted_labels = np.argmax(predicted_result, axis=1)


# 이미지 예측 파일 분류
path_list = []

for i in range(num_classes):
    path_list.append('./dogClassifier/ResNet50_seperate_none_augO/classified_img/' + str(i))

for path in path_list:
  if not os.path.exists(path):
      os.makedirs(path)

print()
for n in range(len(predicted_labels)):
    cv2.imwrite(path_list[predicted_labels[n]] + '/' + list_file_name_valid[n], cv2.cvtColor(images_valid[n] * 255, cv2.COLOR_RGB2BGR))


labels_valid = np.argmax(labels_valid, axis=1)

confusion_matrix = confusion_matrix(labels_valid, predicted_labels)
print("\nconfusion matrix")
print(confusion_matrix)


print("\ntest set class별 개수 확인")
num_check = [0 for i in range(num_classes)]

for i in range(len(labels_valid)):
    num_check[labels_valid[i]] += 1

for i in range(num_classes):
    print("{} 개수: {}".format(i, num_check[i]))


print("\n정확도")
class_accuracy = [0 for i in range(num_classes)]

for i in range(num_classes):
    class_accuracy[i] = confusion_matrix[i][i]/num_check[i]
    print("{} 정확도: {}".format(i, class_accuracy[i]))


# test set에서 predict한 class와 실제 class가 같지 않으면 rename.
# predict된 폴더에 있는 이미지 파일명 뒤에 '_실제class'를 덧붙임
# dir_url = './dogClassifier/ResNet50_seperate_none_augO/classified_img/'
# files = os.listdir(dir_url)
#
#
# for i in range(len(labels_valid)):
#     if predicted_labels[i] != labels_valid[i]:
#         temp_url = dir_url + files[predicted_labels[i]]
#         os.rename(temp_url + '/' + list_file_name_valid[i], temp_url + '/' + list_file_name_valid[i].split('.')[0] + '_' + str(labels_valid[i]) + '.jpg')



# ans = input("예측을 진행하시겠습니까?(y/n): ")
# if ans == 'y':
#     input_img = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                           filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
#     input_img = cv2.imread(input_img, cv2.IMREAD_COLOR)
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     input_img = cv2.resize(input_img, dsize=(224, 224))
#     input_img = input_img.astype("float32") / 255   # (244, 244, 3)
#     input_img = np.expand_dims(input_img, 0)
#     # print(input_img.shape)  # (1, 244, 244, 3)
#     result = model.predict(input_img)
#     # print(predicted_result)
#     # print(predicted_result.shape)   # (1, 10)
#     result = result[0].argmax()
#     print(f'{dict_dogs[result]}입니다.')

    # print(type(input_img))
    # cv2.imshow('img', input_img)
    # cv2.waitKey(1000)

    '''
    dict_dog = {0:'닥스훈트'}
    '''
    # pred = model.predict(input_img) # input_img : (1,224,224,3) / pred : (1,10) -> pred[0] : (10,) -> pred[0].argmax(-1) : (1,) -> {0,1,2,3,4,5,6,7,8,9}
    # print(dict_dog[pred[0].argmax(-1)])


#
# MODEL_SAVE_FOLDER_PATH = './dogClassifier/ResNet50_Pretrained_n(1-1)_7class/model/'
# if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
#     os.mkdir(MODEL_SAVE_FOLDER_PATH)
#
# checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_FOLDER_PATH+'checkpoint1_1.h5', monitor='val_loss', verbose=1, save_best_only=True)
#
# # def scheduler(epoch, lr):
# #     if epoch < 10:
# #         return lr
# #     else:
# #         return lr * math.exp(-0.1)
#
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
# #
# # print("training_set_img shape: ", training_set_img.shape)
# # print("training_set_label shape: ", training_set_label.shape)
# # print("validation_set_img shape: ", validation_set_img.shape)
# # print("validation_set_label shape: ", validation_set_label.shape)
# history = model.fit(training_set_img, training_set_label, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, reduce_lr])
# # history = model.fit_generator(train_dg.flow(training_set_img, training_set_label, batch_size=batch_size),
# #                               steps_per_epoch=math.ceil(training_set_label.shape[0]/batch_size),
# #                               epochs=epochs,
# #                               validation_data=(validation_set_img, validation_set_label),
# #                               callbacks=[checkpoint, reduce_lr])
# # history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=batch_size, epochs=epochs, callbacks=[checkpoint, reduce_lr])
# # MODEL_SAVE_FOLDER_PATH = "C:/Users/sylee/PycharmProjects/pythonProject1/dogClassifier/model/"
# # history = model.load_weights(MODEL_SAVE_FOLDER_PATH+'checkpoint1_1.h5')
#
# ## evaluate
# # 모델 평가: 준비된 test set으로 학습한 모델을 평가한다.
# print("test_set_img1 shape: ", test_set_img.shape)
# print("test_set_label1 shape: ", test_set_label.shape)
# score = model.evaluate(test_set_img, test_set_label, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
#
# ## predict
# predicted_result = model.predict(test_set_img)
# predicted_labels = np.argmax(predicted_result, axis=1)
#
# # 이미지 예측 파일 분류
# path_list = []
#
# for i in range(num_classes):
#     # path_list.append('./dogClassifier/classified_img/ResNet50_Pretrained_y(1-2)/' + str(i))
#     path_list.append('./dogClassifier/ResNet50_Pretrained_n(1-1)_7class/classified_img/' + str(i))
#
#
# for path in path_list:
#   if not os.path.exists(path):
#       os.makedirs(path)
#
# for n in range(len(predicted_labels)):
#     cv2.imwrite(path_list[predicted_labels[n]] + '/' + test_set_fname[n], test_set_img[n] * 255)
#
#
#
# test_set_label = np.argmax(test_set_label, axis=1)
#
# confusion_matrix = confusion_matrix(test_set_label, predicted_labels)
# print("\nconfusion matrix")
# print(confusion_matrix)
#
#
# print("\ntest set class별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(test_set_label)):
#     num_check[test_set_label[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))
#
#
# print("\n정확도")
# class_accuracy = [0 for i in range(num_classes)]
#
# for i in range(num_classes):
#     class_accuracy[i] = confusion_matrix[i][i]/num_check[i]
#     print("{} 정확도: {}".format(i, class_accuracy[i]))
#
#
# # test set에서 predict한 class와 실제 class가 같지 않으면 rename.
# # predict된 폴더에 있는 이미지 파일명 뒤에 '_실제class'를 덧붙임
# dir_url = './dogClassifier/ResNet50_Pretrained_n(1-1)_7class/classified_img/'
# files = os.listdir(dir_url)
#
# for i in range(len(test_set_label)):
#     if predicted_labels[i] != test_set_label[i]:
#         temp_url = dir_url + files[predicted_labels[i]]
#         os.rename(temp_url + '/' + test_set_fname[i], temp_url + '/' + test_set_fname[i].split('.')[0] + '_' + str(test_set_label[i]) + '.jpg')



# print("\n\n\n\nResNet50 - weights=None")
#
# ## ResNet50 - weights=None
# model = ResNet50(weights=None, input_tensor=model_input, include_top=False) # include_top=False => GAP전까지
# x = layers.GlobalAvgPool2D()(model.output)
# x = layers.Dense(num_classes, activation='softmax')(x)
#
#
# model = models.Model(model_input, x)
#
# model.summary()
#
# # train the model
# # 모델 훈련
# batch_size = 16
# epochs = 30 # 훈련 횟수. 전체 데이터를 15번 사용해서 학습을 하는 것.
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= ["accuracy"]) # sgd
#
#
# train_dg = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=30,
#                                                             width_shift_range=0.2,
#                                                             height_shift_range=0.2,
#                                                             #brightness_range=[0.5, 1.5],
#                                                             shear_range=0.1,
#                                                             zoom_range=0.2,
#                                                             fill_mode='nearest',
#                                                             cval=0.0,
#                                                             horizontal_flip=True,
#                                                             #vertical_flip=True,
#                                                             )
#
# MODEL_SAVE_FOLDER_PATH = './dogClassifier/model/'
# if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
#     os.mkdir(MODEL_SAVE_FOLDER_PATH)
#
# checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_FOLDER_PATH+'checkpoint2.h5', monitor='val_loss', verbose=1, save_best_only=True)
#
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
#
# print("training_set_img shape: ", training_set_img.shape)
# print("training_set_label shape: ", training_set_label.shape)
# history2 = model.fit(training_set_img, training_set_label, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, reduce_lr])
# # history2 = model.fit_generator(train_dg.flow(training_set_img, training_set_label, batch_size=batch_size),
# #                               steps_per_epoch=math.ceil(training_set_label.shape[0]/batch_size),
# #                               epochs=epochs,
# #                               validation_data=(validation_set_img, validation_set_label),
# #                               callbacks=[checkpoint, reduce_lr])
#
# ## evaluate
# # 모델 평가: 준비된 test set으로 학습한 모델을 평가한다.
#
# # test_set_img1 shape:  (200, 224, 224, 3)
# # test_set_label1 shape:  (200, 10)
#
# # test_set_img2 shape:  (200, 224, 224, 3)
# # test_set_label2 shape:  (200,
# print("test_set_img2 shape: ", test_set_img.shape)
# print("test_set_label2 shape: ", test_set_label.shape)
#
# score = model.evaluate(test_set_img, test_set_label, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
#
# ## predict
# predicted_result = model.predict(test_set_img)
# predicted_labels = np.argmax(predicted_result, axis=1)
#
# # 이미지 예측 파일 분류
# path_list = []
#
# for i in range(num_classes):
#     path_list.append('./dogClassifier/classified_img/ResNet50_None/' + str(i))
#
# for path in path_list:
#   if not os.path.exists(path):
#       os.makedirs(path)
#
# for n in range(len(predicted_labels)):
#     cv2.imwrite(path_list[predicted_labels[n]] + '/' + str(n) + '.jpg', test_set_img[n] * 255)
#
#
# test_set_label = np.argmax(test_set_label, axis=1)
#
# confusion_matrix = confusion_matrix(test_set_label, predicted_labels)
# print("\nconfusion matrix")
# print(confusion_matrix)
#
#
# print("\ntest set class별 개수 확인")
# num_check = [0 for i in range(num_classes)]
#
# for i in range(len(test_set_label)):
#     num_check[test_set_label[i]] += 1
#
# for i in range(num_classes):
#     print("{} 개수: {}".format(i, num_check[i]))
#
#
# print("\n정확도")
# class_accuracy = [0 for i in range(num_classes)]
#
# for i in range(num_classes):
#     class_accuracy[i] = confusion_matrix[i][i]/num_check[i]
#     print("{} 정확도: {}".format(i, class_accuracy[i]))
#
#
# # test set에서 predict한 class와 실제 class가 같지 않으면 rename.
# # predict된 폴더에 있는 이미지 파일명 뒤에 '_실제class'를 덧붙임
# dir_url = './dogClassifier/classified_img/ResNet50_None/'
# files = os.listdir(dir_url)
#
# for i in range(len(test_set_label)):
#     if predicted_labels[i] != test_set_label[i]:
#         temp_url = dir_url + files[predicted_labels[i]]
#         os.rename(temp_url + '/' + test_set_fname[i], temp_url + '/' + test_set_fname[i].split('.')[0] + '_' + str(test_set_label[i]) + '.jpg')
#

'''
grid = (1, 2)

plt.subplot2grid(grid, (0, 0), rowspan=1, colspan=1)
y_loss = history.history['loss']
y_val_loss = history.history['val_loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, color='steelblue', label='loss')
plt.plot(x_len, y_val_loss, color='slategray', label='val_loss')
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True, color='gray', alpha=0.5, linestyle='--')
plt.tick_params(axis='both', direction='out', length=3)
plt.legend()

plt.subplot2grid(grid, (0, 1), rowspan=1, colspan=2)
y_accuracy = history.history['accuracy']
y_val_accuracy = history.history['val_accuracy']
plt.plot(x_len, y_accuracy, color='palevioletred', label='accuracy')
plt.plot(x_len, y_val_accuracy, color='mediumorchid', label='val_accuracy')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid(True, color='gray', alpha=0.5, linestyle='--')
plt.tick_params(axis='both', direction='out', length=3)
plt.legend()

plt.show()
'''

# grid = (1, 2)
#
# plt.subplot2grid(grid, (0, 0), rowspan=1, colspan=1)
# y_loss1 = history1.history['loss']
# y_val_loss1 = history1.history['val_loss']
# x_len = np.arange(len(y_loss1))
# plt.plot(x_len, y_loss1, color='steelblue', label='loss')
# plt.plot(x_len, y_val_loss1, color='slategray', label='val_loss')
# plt.title("ResNet50 - weights='imagenet'")
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid(True, color='gray', alpha=0.5, linestyle='--')
# plt.tick_params(axis='both', direction='out', length=3)
# plt.legend()
#
# plt.subplot2grid(grid, (0, 1), rowspan=1, colspan=2)
# y_loss2 = history2.history['loss']
# y_val_loss2 = history2.history['val_loss']
# plt.plot(x_len, y_loss2, color='palevioletred', label='loss')
# plt.plot(x_len, y_val_loss2, color='mediumorchid', label='val_loss')
# plt.title("ResNet50 - weights=None")
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid(True, color='gray', alpha=0.5, linestyle='--')
# plt.tick_params(axis='both', direction='out', length=3)
# plt.legend()
#
# plt.show()





# ## CNN에서 정확도를 높이는 방법
# ## 데이터 확장
# # 입력 이미지(훈련 이미지)를 알고리즘을 동원해 인위적으로 확장한다.
# # 입력 이미지를 회전하거나 세로로 이동하는 등 미세한 변화를 주어 이미지 개수를 늘리는 것.
# # 데이터가 몇 개 없을 때 효율적인 방법
# # 다양한 방법으로 이미지를 확장할 수 있다. 예를 들어, 이미지 일부를 잘라내는 crop이나 좌우를 뒤집는 flip.
# # 일반적인 이미지에는 밝기 등의 외형 변화나 확대, 축소 등의 스케일 변화도 효과적이다.
# # 데이터 확장을 동원해 훈련 이미지의 개수를 늘릴 수 있다면 딥러닝의 인식 수준을 개선할 수 있다.
# # model.fit_generator
# # image generator
#
#
# ## VGG16
# # VGG: Visual Geometry Group    16: 16 layers
# # 신경망 모델의 깊이(레이어 수)에 따라 VGG19 또는 VGG13이 될 수도 있다.
#
# # 어떻게 16-19 layer와 같이 깊은 신경망 모델의 학습을 성공했을까? => 모든 convolutional layer에서 3*3 필터를 사용했기 때문
# # 왜 모든 convolutional layer에서 3*3 필터만 사용했을까?
# # =>    1. 결정 함수의 비선형성 증가 : 레이어가 증가함에 따라 비선형성이 증가하게 되고 이것은 모델의 특징 식별성 증가로 이어진다.
# #       2. 학습 파라미터 수의 감소 : Convolutional Network 구조를 학습할 때, 학습 대상인 가중치(weight)는 필터의 크기에 해당한다.
# #                                 7*7 filter 1개에 대한 학습 파라미터 수: 49 (7*7)
# #                                 3*3 filter 3개에 대한 학습 파라미터 수: 27 (3*3*3)
# #                                 => 파라미터 수가 크게 감소
#
# ## VGG-16 architecture의 구성
# # 13 convolution layers + 3 fully-connected layers
# # 3*3 convolution filters
# # stride: 1 & padding: 1
# # 2*2 max pooling (stride: 2)
# # ReLU
# # VGG16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] + ['AA', 4096, 4096, 1000]
#
#
# ## Augmentation
# # 모델이 적인 이미지에서 최대한 많은 정보를 뽑아내서 학습할 수 있도록 이미지를 사용할 때마다 임의로 변형을 가함으로써 마치 훨씬 더 많은 이미지를 보고 공부하는 것과 같은 학습 효과를 낸다.
# # 이를 통해, 과적합(overfitting), 즉 모델이 학습 데이터에만 맞춰지는 것을 방지하고, 새로운 이미지도 잘 분류할 수 있게 된다.
#
# # 이런 전처리 과정을 돕기 위해 케라스는 ImageDataGenerator 클래스를 제공한다.
# ## ImageDataGenerator
# # 1. 학습 도중에 이미지에 임의 변형 및 정규화 적용
# # 2. 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성
# #       generator를 생설할 때, flow(data, labels), flow_from_directory(directory) 두 가지 함수를 사용
# #       fit_generator, evalute_generator 함수를 이용해서 generator로 이미지를 불러와서 모델을 학습시킬 수 있다.
#
#
# # 모델 성능을 평가할 때에는 이미지 원본을 사용한다.
