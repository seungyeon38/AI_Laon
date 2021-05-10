# setup
import numpy as np
# from keras import models
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# prediction
import os
import cv2

from tensorflow.keras.applications.resnet50 import ResNet50

# 1. 데이터 처리하는 부분은 뭐가 어떻게 변하게 되는지를
#    데이터가 어떻게 변하는지는 shape도 같이 함께 적어주세요.
# 2. 학습 이론에 관련된 부분은 뭘 위한 작업이고 어떤 효과가 있는지를

# def loss_plot(history):
#     plt.plot(history.history['loss']), label='Train loss')
#     pl


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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# prepare data
# Model / data parameters
# target class (0~9)
num_classes = 10
# 이미지데이터: 28*28 pixel, 흑백: 1
input_shape = (28, 28, 1)

np.random.seed(10)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train: 60000개의 28*28 크기의 이미지
# y_train: x_train의 60000개에 대한 값(0~9)이 담겨 있는 target 값
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# Scale images to the [0, 1] range
### 정규화(normalization) : 값 전체의 범위가 어떤 형태의 범위 안에 항상 들어가도록 preprocessing하는 방법
# 기본적으로 정규화를 하는 이유는 학습을 더 빨리 하기 위해서 또는 local optimum 문제에 빠지는 가능성을 줄이기 위해서 사용한다.
x_train = x_train.astype("float32") / 255 # (60000, 28, 28) 28*28 픽셀 데이터 60000개
x_test = x_test.astype("float32") / 255 # (10000, 28, 28) 28*28 픽셀 데이터 10000개
# 255로 나누어주려면 먼저 값을 실수형으로 바꾼 후에 나누어야 함.


# 정규화: 이 방법보다는 정규분포?방법을 많이 사용할 것
# 평균 빼고 나누는 방법?
# standardization: (오차값-평균)/표준편차

# Make sure images have shape (28, 28, 1)
### expand_dims(arr, axis) : input array. new axis가 어디에 insert될지
# -1을 많이 쓰게 될 것. -1은 맨 뒤를 얘기한다.
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)  # (60000, 28, 28, 1) # transpose, reshape, stack, np.array(list_image).shape -> (60000, 28*28) ->
# print("y_train shape:", y_train.shape)  # (60000,)
print(x_train.shape[0], "train samples") # 데이터 개수
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
# Keras의 기능인 to_categorical을 통해 정수형 타겟(integer target)을 범주형 타겟(categorical target)으로 변환할 수 있다.
# ont-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print("y_train shape:", y_train.shape) # (60000, 10)
# print("y_test shape:", y_test.shape) # (10000, 10)
# print("y_train: ", y_train)
# print("y_test: ", y_test)

# CNN은 convolution layer와 max pooling 레이어를 반복적으로 stack을 쌓는 특징 추출(feature extraction)부분과
# fully connected layer를 구성하고 마지막 출력층에 softmax를 적용한 분류 부분으로 나뉜다.

# filter, stride, padding을 조절하여 특징 추출(Feature Extraction) 부분의 입력과 출력 크기를 계산하고 맞추는 작업이 중요하다.

# 학습 파라미터 = 입력 채널수 * 필터 사이즈(필터폭 * 필터 높이) * 출력 채널

# 하나의 convolution layer에 크기가 같은(weight은 다른?) 여러 개의 필터를 적용할 수 있다.
# feature map에는 필터 개수 만큼의 채널이 만들어진다.
# 입력데이터에 적용한 필터의 개수는 출력데이터인 feature map의 채널이 된다.
# convolution layer의 입력 데이터를 필터가 순회하며 합성곱을 통해서 만든 출력을 feature map 또는 activation map이라고 한다.
# feature map은 합성곱 계산으로 만들어진 행렬(입력데이터는 채널 수와 상관없이 필터 별로 1개의 feature map이 만들어진다.)
# activation map은 feature map 행렬에 활성 함수를 적용한 결과(convolution layer의 최종 출력 결과)

### 활성화 함수: 입력값을 non-linear한 방식으로 출력값을 도출하기 위해 사용. linear system을 non-linear한 system으로 바꿀 수 있게 된다.
# 왜 linear한 방식은 안 쓰는지?
# 어차피 같아지기 때문에
# y = f(f(f(x)))
# f(x) = c*x
# y = c*c*c*x
# a = c^3
# y = a*x

# 활성화 함수를 사용하는 두 가지 이유
# 1. 모델 layer들 사이에 존재하면서 학습을 도와주고, 2. 모델의 가장 마지막 출력에 관여하면서 출력의 형태를 변화시킴

# build the model
# 모델 구성
model_input = keras.Input(shape=input_shape)

model = ResNet50(weights=None, input_tensor=model_input, include_top=False)
x = layers.GlobalAvgPool2D()(model.output)
x = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(model_input, x)

#
# model = keras.Sequential(
#     [
#         # 입력층 설계
#         keras.Input(shape=input_shape),
#         # convolution layer: 입력 데이터에 필터를 적용시키고 활성화 함수를 반영. 필수.
#         # activation function이 sigmoid가 아닌 ReLU인 이유
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),   # (26, 26, 32)
#         # pooling layer: convolution layer의 출력 데이터를 입력으로 받아서 출력 데이터(activation map)의 크기를 줄이거나 특정 데이터를 강조하는 용도로 씀. 선택적
#         layers.MaxPooling2D(pool_size=(2, 2)),  # (13, 13, 32)
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),   # (11, 11, 64)
#         layers.MaxPooling2D(pool_size=(2, 2)),  # (5, 5, 64) # (10, 10, 64)
#
#         # CNN에서 convolution layer와 pooling layer를 반복적으로 거치면 주요 특징만 추출된다.
#         # 추출된 주요 특징은 2차원 데이터로 이루어져 있지만, dense와 같이 분류를 위한 학습 레이어에서는 1차원 데이터로 바꿔서 학습이 돼야 한다.
#
# ## x1
#         layers.Flatten(), # (5*5*64,)
#         ## GlobalAveragePooling2D # (5, 5, 64) -> (1, 1, 64) ->(또는) (64,)
#         ### flatten layer:  이미지의 특징 추출 부분과 이미지 분류 부분 사이에 이미지 형태의 데이터를 배열 형태로 만드는 layer.(2차원 -> 1차원)
#         #                   CNN의 data type을 fully connected neural network 형태로 변경하는 layer.
#         #                   파라미터가 존재하지 않고, 입력 데이터의 shape 변경만 한다.
#
# # flatten보다는 GlobalAveragePooling2D를 상당히 많이 쓸 것. 거의 다 쓸 것.
# # flatten은 해상도나 비율에 따라서 문제가 많기 때문에.
#
# ## W
#         layers.Dropout(0.5),
#         # 네트워크가 과적합되는 경우를 방지하기 위해서 만들어진 layer
#         # overfitting 방지. 학습할 때 임의적으로 몇 개의 노드를 끊어버리고 노드를 줄임. 보통 0.5를 많이 쓴다.
#         # 학습할 동안만 dropout하는 것. 실제로는 모든 걸 불러와야 한다.
#
# ## x1*W
# #  x2
#        layers.Dense(num_classes, activation="softmax"),
#         # softmax:  다중 클래스 분류 문제(multinomial classification)에서 출력층에 주로 씀.
#         #           입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수
#         # 1.0~1 사이로, 2.합이 1로. -> 각각이 확률로 볼 수 있다.
#     ]
# )

# Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
# 첫번째 인자 : 컨볼루션 필터의 수
# 두번째 인자 : 컨볼루션 커널(필터)의 (행, 열)
# padding : 경계 처리 방법
# ‘valid’ : 유효한 영역만 출력이 됨. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작다.
# ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일하다.
# input_shape : 샘플 수를 제외한 입력 형태를 정의한다. 모델에서 첫 레이어일 때만 정의하면 된다.
# (행, 열, 채널 수)로 정의한다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정한다.
# activation : 활성화 함수 설정한다.
# ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나온다.
# ‘relu’ : 은익층에 주로 쓰인다.
# ‘sigmoid’ : 이진 분류 문제에서 출력층에 주로 쓰인다.
# ‘softmax’ : 다중 클래스 분류 문제에서 출력층에 주로 쓰인다.

# filter: 이미지의 특징을 찾아내기 위한 공용 파라미터. kernel이라고도 한다.
# CNN에서 학습의 대상은 필터 파라미터

model.summary()

# train the model
# 모델 훈련
batch_size = 128
epochs = 30 # 훈련 횟수. 전체 데이터를 15번 사용해서 학습을 하는 것.

# 다루어야 할 데이터가 너무 많기도 하고(메모리가 부족하기도 하고) 한 번의 계산으로 최적화된 값을 찾는 것이 힘들다.
# 따라서, 머신 러닝에서 최적화를 할 때는 일반적으로 여러 번 학습 과정을 거친다. 또한 한 번의 학습 과정 역시 사용하는 데이터를 나누는 방식으로 세분화시킨다.
# 이때, epoch, batch size, iteration라는 개념이 필요하다.
# 한 번의 epoch는 인공 신경망에서 전체 데이터 셋에 대해 forward pass/ backward pass 과정을 거친 것을 말한다. 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
# 모델을 만들 때 적절한 epoch값을 설정해야만 underfitting과 overfitting을 방지할 수 있다. epoch 값이 너무 작다면 underfitting이, 너무 크다면 overfitting이 발생할 확률이 높은 것.

# 한 번의 batch마다 주는 데이터 샘플의 size. 여기서 batch(보통 mini batch라고 표현)는 나눠진 데이터 셋을 뜻하며 iteration은 epoch를 나누어서 실행하는 횟수라고 생각하면 된다.
# 메모리의 한계와 속도 저하 때문에 대부분의 경우에는 한 번의 epoch에서 모든 데이터를 한꺼번에 집어넣을 수는 없다.
# 그래서 데이터를 나누어서 주게 되는데 이때 몇 번 나누어서 주는가를 iteration, 각 iteration마다 주는 데이터 사이즈를 batch size라고 한다.

# epoch - 동일한 학습 데이터라고 하더라도 여러번 학습할수록 학습 효과가 커진다. 하지만, 너무 많이 하면 overfitting이 발생한다.
# batch size에 따라서 학습 결과에 큰 차이가 있다.
# 사람이 100문제를 풀고 한 번에 채점하는 것과, 1문제를 풀고 채점한 다음, 다음 문제를 푸는 것과 같다.
# batch size값이 크면 클수록 여러 데이터를 기억하고 있어야 하기 때문에 메모리가 커야 한다. 그 대신 학습 시간이 빨라진다.
### batch size값이 작으면 학습은 꼼꼼하게 이루어질 수 있지만 학습 시간이 많이 걸린다.
# batch normalization 공부하기. 굉장히 많이 쓰고, 굉장히 중요.

# 1 epoch: 모든 데이터 셋을 한 번 학습
# 1 iteration: 1회 학습
# minibatch: 데이터 셋을 batch size 크기로 쪼개서 학습

# 모델 학습과정 설정: 모델 학습 전에 학습에 대한 설정을 수행한다. / 손실 함수 및 최적화 방법을 정의.
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= ["accuracy"]) # sgd
### 손실 함수(목적 함수, 최적화 점수 함수) : 하나의 모델을 컴파일하기 위해 필요한 두 개의 매개변수 중 하나.
# 손실 함수 categorical_crossentropy의 경우 사용되는 타겟들은 범주 형식 (categorical format)을 따라야 한다.
# 예를 들어 10개의 클래스(범주) 중 하나에 속하는 데이터에 대하여 각 샘플은 타겟 클래스에 해당하는 하나의 인덱스만 1의 값을 가지고 이외의 값들은 모두 0이어야 한다.
# Keras의 기능인 to_categorical을 통해 정수형 타겟(integer target)을 범주형 타겟(categorical target)으로 변환할 수 있다.

# optimization: 학습속도를 빠르고 안정적이게 하는 것 # 이건 adam에 대한 설명이지 optimizer에 관한 설명이 아님(틀림)

## loss function: 실제값과 예측값의 차이(loss, cost)를 수치화해주는 함수
# 오차가 클수록 손실 함수의 값이 크고, 오차가 작을수록 손실 함수의 값이 작아진다.
# 손실 함수의 값을 최소화하는 W, b를 찾아가는 것이 목표
# regression(회귀): 평균제곱오차(MSE) / 분류: cross entropy(이진분류: binary_crossentropy, 다중분류: categorical_crossentropy)

## optimizer
# 손실함수를 줄여나가면서 학습하는 방법은 어떤 optimizer를 사용하느냐에 따라 달라진다.
# gradient descent, batch gradient descent, SGD(stochastic gradient descent), mini-batch gradient descent, momentum, adam

## loss function: 훈련하는 동안 모델의 오차를 측정. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 한다.
## optimizer: 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정.
## metrics: 훈련 단계와 테스트 단계를 모니터링하기 위해 사용.

## stochastic gradient descent(sgd) : gradient descent를 전체 데이터(batch)가 아닌 일부 데이터의 모음(mini-batch)를 사용하는 방법.
# mini-batch를 사용하기 때문에 다소 부정확할 수는 있지만 계산 속도가 훨씬 빠르기 때문에 같은 시간에 더 많은 step을 나아갈 수 있음.
# local minima에 빠지지 않고 global minima에 수렴할 가능성이 더 높음.
# stochastic grandient는 실제 gradient의 추정값이며 이것은 mini batch의 크기 N이 커질수록 더 정확한 추정값을 가지게 된다.
# mini batch를 뽑아서 연산을 수행하기 때문에 최신 컴퓨팅 플랫폼에 의하여 병렬적인 연산 수행이 가능하여 더 효율적이다.
# neural network을 학습하기 위한 하이퍼 파라미터들의 초기값 설정을 굉장히 신중하게 해줘야 한다는 것이 문제.

# whitening: 기본적으로 들어오는 input의 feature들을 uncorrelated하게 만들어주고, 각각의 variance를 1로 만들어주는 작업
# 계산량이 많고 일부 parameter들의 영향이 무시된다는 것이 단점.

# learning rate

# 분류 문제에서 주로 사용하는 활성화함수와 로스.
# 분류 문제에서는 MSE(mean square error) loss보다 CE(crossentropy) loss가 더 빨리 수렴한다는 사실이 알려져있다.
# 따라서 multi class에서 하나의 클래스를 구분할 때 softmax와 CE loss의 조합을 많이 사용한다.

# metric: 평가 기준. metrics인자로 여러 개의 평가 기준을 지정할 수 있다. 모델의 학습에는 영향을 미치지 않지만, 학습 과정 중에 제대로 학습되고 있는지 살펴볼 수 있다.
# 분류 문제에서 클래스별로 확인할 때는 정밀도와 재현율을 파악하는 것이 도움이 된다.


# train_dg = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=0.2,
#                                                             width_shift_range=0.2,
#                                                             height_shift_range=0.2,
#                                                             # brightness_range=[0.5, 1.5],
#                                                             shear_range=0.2,
#                                                             zoom_range=0.2,
#                                                             fill_mode='constant',
#                                                             cval=0.0, # fill_mode = "constant"일 때 씀
#                                                             # horizontal_flip=True,
#                                                             # vertical_flip=True,
#                                                             )

# MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_SAVE_FOLDER_PATH = './cifar10/ResNet50_none/model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_FOLDER_PATH+'checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True)
#, save_weights_only=True) # 이거 하니까 안됨,,,?
# 모델 학습: training set을 이용해서 구성한 모델로 학습시킴. / fit(입력데이터, 결과(label값)데이터, 한 번에 학습할 때 사용하는 데이터 개수, 학습 데이터 반복 횟수)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# keras.callbacks.LearningRateScheduler(schedule, verbose=0)
# 매 epoch의 처음마다, callback은 __init__에서 제공되는 schedule 함수로부터 업데이트된 learning rate 값을 가져오고, optimizer에 업데이트된 learning rate를 적용한다.
# schedule : epoch index를 입력 (정수, 0으로부터 index)으로 사용하고 현재 학습 속도(float)를 가져와서 새로운 학습 속도를 출력(float)으로 반환하는 함수입니다.

# def scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * math.exp(-0.1)

# learningRate = keras.callbacks.LearningRateScheduler(scheduler)


# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, learningRate])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, reduce_lr])


# history = model.fit_generator(train_dg.flow(x_train, y_train, batch_size=batch_size),
#                               steps_per_epoch=math.ceil(y_train.shape[0]/batch_size),
#                               epochs=epochs,
#                               validation_data=(x_test, y_test),
#                               callbacks=[checkpoint, reduce_lr])

# test set을 validation set으로 써도 되는건지 모르겠음

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint])

# print(MODEL_SAVE_FOLDER_PATH+'checkpoint.h5')
# model.load_weights(MODEL_SAVE_FOLDER_PATH+'checkpoint.h5')

# print(history.history.keys()) # loss, accuracy, val_loss, val_accuracy

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
#
# # title, label, grid, tick, legend
# # tick: 그래프의 축에 간격을 구분하기 위해 표시하는 눈금
# # xticks(), yticks(), tick_params() 함수
# # tick_params() : 눈금의 스타일을 다양하게 설정 가능
#
# # plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
# # grid() 첫번째 파라미터를 true로 설정하면 그래프에 grid가 표시된다.
# # axis='y'로 설정하면 가로 방향의 그리드만 표시된다.
# # alpha는 그리드의 투명도를 설정. 0으로 설정하면 투명하게, 1은 불투명하게 표시된다.
# # linestyle을 이용해서 선의 스타일을 대쉬(dashed)로 설정한다.
#
# # plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
# # tick_params() 그래프의 틱과 관련된 설정을 할 수 있다.
# # axis='both'로 설정하면 x, y축의 틱이 모두 적용된다.
# # direction='in'으로 틱의 방향을 그래프 안쪽으로 설정.
# # 틱의 길이(length)를 3으로, 틱과 레이블의 거리(pad)를 6으로 설정.
# # 틱 레이블의 크기(labelsize)를 14로 설정.
#
#
# epoch, loss 간 관계
# y_val_loss = history.history['val_loss']
# 이거 대신에 불러와야 한다.
# x_len = np.arange(len(y_val_loss)) # array 자료형이라 계산하는데 편리
# plt.plot(x_len, y_val_loss, color='slategray', label='val_loss')
# plt.title('loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid(True, color='gray', alpha=0.5, linestyle='--')
# plt.tick_params(axis='both', direction='out', length=3)
# plt.legend()
#
# plt.show()
#
# # 그래프 learning rate, loss function
# # epoch, loss간 관계
# # epoch마다 loss, accuracy 그래프
#
# # history의 값에 있는  loss 변화 그래프로 출력 (plt.plot 활용)
# # history[][] 두 개였던 것 같음.
#
# # ML을 나눠서 평가해야 한다. test set은 사용하면 안되고 숨어져있다고 생각하고 training set만 가지고 판단해야 한다.
# # 통상적으로 training, test로 나뉘고 training data를 validation으로도 나눌 때도 있다.
# # validation_split: 학습 시 데이터를 일부 나눠서 validation으로 사용할 비율을 의미한다.
#
# # preds = model.predict(x_test) # (10000, 10)
# # 이 코드에 predict는 없는데 추가해보기.
#
# evaluate the trained model
# 모델 평가: 준비된 test set으로 학습한 모델을 평가한다.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# ## predict
# # cv2.imwrite(filename, image) 형식
#
predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)
# 이미지 예측 파일 분류
# path_list = ['.cifar10_img/classfied_img/0', '.cifar10_img/classfied_img/1', '.cifar10_img/classfied_img/2', '.cifar10_img/classfied_img/3', '.cifar10_img/classfied_img/4', '.cifar10_img/classfied_img/5', '.cifar10_img/classfied_img/6', '.cifar10_img/classfied_img/7', '.cifar10_img/classfied_img/8', '.cifar10_img/classfied_img/9']
path_list = []

for i in range(num_classes):
    path_list.append('./cifar10/ResNet50_none/classified_img/' + str(i))

for path in path_list:
  if not os.path.exists(path):
      os.makedirs(path)

for n in range(len(predicted_labels)):
    cv2.imwrite(path_list[predicted_labels[n]]+ '/' + str(n) + '.jpg', x_test[n]*255)
    # cv2.imwrite(path_list[predicted_labels[n]]+ '/' + str(n) + '.jpg', x_test[n].reshape(28, 28), [cv2.IMWRITE_JPEG_QUALITY, 100])

y_test = np.argmax(y_test, axis=1)

confusion_matrix = confusion_matrix(y_test, predicted_labels)
print("\nconfusion matrix")
print(confusion_matrix)
# print(y_test)
## learning rate scheduling
# callback함수: 사람이 아닌 시스템(컴퓨터, 혹은 프로그램)이 내부적으로 호출하는 함수
# A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).
# keras에서는 tensorboard 등의 callback 함수를 제공한다. 사용자가 자체적으로 callback함수를 만들어서 매 epoch마다 일정 부분 학습 과정에 관여할 수 있다.
# 딥러닝 모델을 학습할 때 학습률(learning rate)를 감소시키는 방법이 자주 사용된다. local minima를 탈출해서 더 낮은 loss로 수렴하기 위함.
# keras에서 제공하는 loss 중 일부는 입력값으로 학습률 감소율을 정해줄 수 있지만, 그것만으로는 부족하다.
# 학습률 감소를 지수적으로 하는 경우도 있고, 계단 모양으로 뚝뚝 떨어뜨리는 경우도 있다.
# 이렇듯 자기 입맛대로 학습률을 조정하기 위해 제공되는 callback함수가 있다ModelCheckpoint. LearningRateScheduler.
# def schedule(): 처럼 학습률을 관리하는 함수를 만든다. 그 함수를 LearningRateScheduler(schedule)처럼 넣어주고 model.fit()에 사용하면 된다.

# learning rate schedules: adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule하기 위해
# Common learning rate schedules include time-based decay, step decay and exponential decay.

## fine tuning
# 기존에 학습되어져있는 모델을 기반으로 아키텍쳐를 새로운 목적(나의 이미지 데이터에 맞게)변형하고 이미 학습된 모델 Weights로부터 학습을 업데이트하는 방법을 말한다.
# - 모델의 파라미터를 미세하게 조정하는 행위 (특히 딥러닝에서는 이미 존재하는 모델에 추가 데이터를 투입하여 파라미터를 업데이트하는 것을 말한다.)
# fine tuning을 했다고 말하려면 기존에 학습이 된 레이어에 내 데이터를 추가로 학습시켜 파라미터를 업데이트 해야 한다. 이 때 주의할 점은, 정교해야 한다는 것이다.
# 완전히 랜덤한 초기 파라미터를 쓴다거나 가장 아래쪽의 레이어의 파라미터를 학습해버리면 overfitting이 일어나거나 전체 파라미터가 망가지는 문제가 생기기 때문이다.

# fine tuning: 새로운 데이터를 다시한번 가중치를 세밀하게 조정하도록 학습. 기존 데이터는 기존대로 분류.
# feature extraction: 기존 가중치는 그대로 놔둔 뒤, 새로운 레이어를 추가해서 이를 학습하고 최종 결과를 내게끔 학습.
# joint training: 새로운 데이터를 추가하여 처음부터 다시 시작
# learning without forgetting: 새로운 데이터로 가중치를 세밀하게 조정하되, 기존 데이터 분류 결과 또한 개선 가능(하다고 주장)

# predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
# predict input 샘플에 대한 output 예측을 생성. 계산은 배치 단위로 실행됨.
# x: Numpy 배열(혹은 모델이 다중 인풋을 갖는 경우 Numpy 배열의 리스트) 형태의 input data
# batch_size: 정수. 디폴트: 32
# verbose: 다변 모드. 0 또는 1
# steps: 예측이 한 회 완료되었음을 선언하기까지 단계(심플 배치)의 총 개수. 디폴트 값인 None의 경우 고려되지 않는다.
# callbacks: 학습과 검증 과정에서 적용한 콜백의 리스트

# 반환값: 예측 값의 Numpy 배열

## ModelCheckPoint class
# tf.keras.callbacks.ModelCheckpoint(
#     filepath,
#     monitor="val_loss",
#     verbose=0,
#     save_best_only=False,
#     save_weights_only=False,
#     mode="auto",
#     save_freq="epoch",
#     options=None,
#     **kwargs
# )

## filepath: path to save model file
# named formatting options(epoch의 값, loss에 있는 key들 등)를 포함할 수 있다.

## monitor: monitor할 metric이름
# 보통 Model.compile method에 의해 set됨

## save_best_only가 true이면 model이 best라고 생각될 때만 저장됨. 최신 best model이 안 덮여쓰일 수 있다.
# filepath가 {epoch}같은 fotmatting option들을 포함하지 않으면, filepath는 각 새로운 더 나은 model에 의해 overwritten될 것이다.

## mode: 'auto', 'min', 'max' 중 하나.
# save_best_only=True면, 현재 저장 파일을 overwrite결정이 monitor되고 있는 quantity의 maximization이나 minimization에 의해 만들어질 것이다.

## save_freq: 'epoch' 또는 integer
# epoch이면 callback함수는 매 epoch마다 model을 저장한다.
# integer이면 이만큼의 batch의 끝마다 model을 저장한다.
# default는 epoch


# Keras model이나 model weights를 어떤 간격으로 저장하기 위해서
# model.fit() 모델학습과 함께 사용.
# 지금까지 best performance를 달성한 모델 또는 performance와 상관없이 대 epoch의 끝마다 모델 저장
# best: quantity to monitor. 그게 maximize되거나 minimize되어야 할 것.
# weight만 저장할지, 전체 model을 저장할지


## Example

# model.compile(loss=..., optimizer=...,
#               metrics=['accuracy'])
#
# EPOCHS = 10
# checkpoint_filepath = '/tmp/checkpoint'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)
#
# # Model weights are saved at the end of every epoch, if it's the best seen so far.
# model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
#
# # The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)


# 딥러닝 모델을 만든다 : layer들을 쌓고, activation function 및 loss function등을 잘 정의해주는 것
# 딥러닝 모델을 학습시킨다 : backpropagation을 통해 최적의 weight값을 알아낼 때까지 반복적으로 모델을 트레이닝 시키는 것
# 모델과 학습으로 얻어낸 최적의 weight값을 저장하지 않으면, 다음에 이 모델을 다시 쓰고자 할 떄마다 매번 기나긴 학습의 과정을 거쳐야 한다.
# 그래서 이 모델과 weight을 저장해두고, 다시 쓰고 싶을 때 불러오기만 해서 쓸 수 있도록 하는 방법이 있다.
# keras에서는 모델을 저장하고 불러오는 기능이 있는 라이브러리를 제공한다.
# 이건 modelcheckpoint 관련이 아님

# confusion matrix
# [[ 975    0    1    0    0    1    1    1    1    0]  # 0 980개 중, 맞게 예측한 것이 975개, 2,5,6,7,8로 잘못 예측한 것이 1개씩
#  [   0 1133    1    0    0    0    0    1    0    0]  # 1 1135개 중, 맞게 예측한 것이 1133개, 2,7로 잘못 예측한 것이 1개씩
#  [   1    4 1014    1    1    0    0    9    2    0]  # 2 1032개 중, 맞게 예측한 것이 1014개, 0,3,4로 잘못 예측한 것이 1개씩, 1로 잘못 예측한 것이 4개, 7로 잘못 예측한 것이 9개, 8로 잘못 예측한 것이 2개
#  [   0    0    2  993    0    5    0    8    2    0]
#  [   1    1    0    0  974    0    2    0    1    3]
#  [   2    0    1    4    0  880    2    1    2    0]
#  [   7    3    0    0    1    4  942    0    1    0]
#  [   0    4    8    2    0    0    0 1013    1    0]
#  [   5    1    4    1    3    1    0    6  948    5]
#  [   3    5    1    1    6    5    0    7    2  979]]

#  현재
# [[ 976    0    1    0    0    0    1    1    1    0]
#  [   0 1132    2    0    0    0    0    1    0    0]
#  [   1    1 1020    1    1    0    0    7    1    0]
#  [   0    1    2  999    0    5    0    2    1    0]
#  [   0    0    0    0  977    0    2    0    1    2]
#  [   0    0    0    3    0  888    1    0    0    0]
#  [   1    2    1    0    1    6  945    0    2    0]
#  [   0    1    2    1    0    0    0 1022    1    1]
#  [   2    0    2    0    0    0    1    2  965    2]
#  [   2    2    0    1    3    5    0    5    1  990]]


print("\nclass별 개수 확인")
num_check = [0 for i in range(num_classes)]

for i in range(len(y_test)):
    num_check[y_test[i]] += 1

for i in range(num_classes):
    print("{} 개수: {}".format(i, num_check[i]))


print("\n정확도")
class_accuracy = [0 for i in range(num_classes)]

for i in range(num_classes):
    class_accuracy[i] = confusion_matrix[i][i]/num_check[i]
    print("{} 정확도: {}".format(i, class_accuracy[i]))


# dir_url = 'C:/Users/sylee/PycharmProjects/pythonProject1/classfied_img/'
dir_url = '../cifar10/ResNet50_none/classified_img/'

files = os.listdir(dir_url)

for i in range(len(y_test)):
    if predicted_labels[i] != y_test[i]:
        temp_url = dir_url + files[predicted_labels[i]]
        os.rename(temp_url + '/' + str(i) + '.jpg', temp_url + '/' + str(i) + '_' + str(y_test[i]) + '.jpg') # 이미지가 하나 빠지면 문제가 됨. 겹치는 파일명이 생길거임.



## validation set
# ML 또는 통계에서 기본적인 개념 중 하나
# 모델의 성능을 평가하기 위해서 사용한다. training을 한 후에 만들어진 모형이 잘 예측을 하는지 그 성능을 평가하기 위해서 사용한다.
# training set의 일부를 모델의 성능을 평가하기 위해서 희생하는 것. 하지만 이 희생을 감수하지 못할만큼 data set의 크기가 작다면 cross-validation이라는 방법을 쓰기도 한다.

# 모델의 성능을 평가하면 좋은 것
# 1. test accuracy를 가늠할 수 있다.
# 2. 모델을 튜닝해서 모델의 성능을 높일 수 있다. 예를 들어 overfitting 등을 막을 수 있다.
#   예를 들어 training accuracy는 높은데 validation accuracy는 낮다면 데이터가 training set에 overfitting이 일어났을 가능성을 생각해볼 수 있다.
#   deep learning을 모델을 구축한다면 regularization 과정을 한다거나 epoch을 줄이는 등의 방식으로 overfitting을 막을 수 있다.

# validation test set과의 차이점은 test set은 모델의 최종 성능을 평가하기 위해서 쓰이며, training의 과정에 관여하지 않는 차이가 있다.
# 반면, validation set은 여러 모델 중에서 최종 모델을 선정하기 위한 성능 평가에 관여한다고 보면 된다. 따라서 validation set은 training 과정에 관여하게 된다.
# 즉, validation set은 training 과정에 관여를 하며, training이 된 여러가지 모델 중 가장 좋은 하나의 모델을 고르기 위한 셋이다.
# test set은 모든 training 과정이 완료된 후에 최종적으로 모델의 성능을 평가하기 위한 셋이다.


## hyperparameter
# hyperparameter란 machine learning 학습을 할 때에 더 효과가 좋도록 하는 주 변수가 아닌 자동 설정 되는 변수