# setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# prediction
import os
import cv2
import math
from tqdm import tqdm

from tensorflow.keras.applications.resnet50 import ResNet50


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
input_shape = (128, 128, 3) # (32, 32, 3)에서 (128, 128, 3)으로 resize

np.random.seed(10)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# x_train: 50000개의 32*32 크기의 이미지
# y_train: x_train의 50000개에 대한 값(0~9)이 담겨 있는 target 값

# resize를 하는 method에도 여러 종류가 있고 각각의 특징들이 있다.
# bilinear, bicubic, lanczos
# bilinear: 용량이 더 작아지고 인토딩 속도도 빠르지만 흐릿한 느낌을 줌
# bicubic: 용량, 속도, 선명함에서 중간 정도
# lanczos: 용량도 커지고 인코딩 속도도 느리지만 가장 선명한 화질을 보여줌
# 실제로 직접 해보고 자기에게 적합한 방식을 고르는 것이 중요하다.
# 단지 다운샘플링은 bilinear, 업샘플링은 bicubic이라는 일종의 불문률만 존재.

# resize method : lanczos
# interpolation 속성: INTER_NEAREST, INTER_LINEAR, INTER_LINEAR_EXACT, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4
# INTER_NEAREST: 가장 빠르지만 퀄리티가 떨어져서 잘 쓰이지 않는다.
# INTER_LINEAR: 효율성이 가장 좋고 속도가 빠르고 퀄리티도 적당하다.
# INTER_CUBIC: INTER_LINEAR보다 느리지만 퀄리티는 더 좋다.
# INTER_LANCZOS4: 좀더 복잡해서 오래 걸리지만 퀄리티는 좋다.
# INTER_AREA: 영상 축소시 효과적. 영역적인 정보를 추출해서 결과 영상을 셋팅한다.

# resize to 128
x_train_resized = []
x_test_resized = []

# print("x_train: ", x_train)
for img in tqdm(x_train):
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LANCZOS4) # ResNet50 모델이 224*224를 타겟으로 만들어진 모델 
    x_train_resized.append(img)
# print("shape of x_train: ", x_train.shape)  # (50000, 32, 32, 3)
# print("shape of y_train: ", y_train.shape)  # (50000, 1)

x_train_resized = np.asarray(x_train_resized)

for img in tqdm(x_test):
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LANCZOS4)
    x_test_resized.append(img)

x_test_resized = np.asarray(x_test_resized)


# print("shape of x_train_resized: ", x_train_resized.shape)  # (50000, 128, 128, 3)


# Scale images to the [0, 1] range
### 정규화(normalization) : 값 전체의 범위가 어떤 형태의 범위 안에 항상 들어가도록 preprocessing하는 방법
# 기본적으로 정규화를 하는 이유는 학습을 더 빨리 하기 위해서 또는 local optimum 문제에 빠지는 가능성을 줄이기 위해서 사용한다.
x_train_resized = x_train_resized.astype("float32") / 255 # (50000, 32, 32) 32*32 픽셀 데이터 50000개
x_test_resized = x_test_resized.astype("float32") / 255 # (10000, 32, 32) 32*32 픽셀 데이터 10000개
# 정규화: 이 방법보다는 정규분포?방법을 많이 사용할 것
# 평균 빼고 나누는 방법?
# standardization: (오차값-평균)/표준편차
# Make sure images have shape (28, 28, 1)
### expand_dims(arr, axis) : input array. new axis가 어디에 insert될지
# -1을 많이 쓰게 될 것. -1은 맨 뒤를 얘기한다.
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# # x_train = np.expand_dims(x_train, -1)
# # x_test = np.expand_dims(x_test, -1)
# print(x_train.shape[0], "train samples") # 데이터 개수
# print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
# Keras의 기능인 to_categorical을 통해 정수형 타겟(integer target)을 범주형 타겟(categorical target)으로 변환할 수 있다.
# ont-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

'''
// fine tuning의 개념 
model.layers[:-1] -> freezing
model.fit -> small lr/epoch
model.layers[:-1] -> unfreezing
model.fit
'''

model = models.Model(model_input, x, name='resnet50_none_augO')
# model = models.Model(model_input, x)

model.summary()

# train the model
# 모델 훈련
batch_size = 64 # 128
epochs = 100

# 모델 학습과정 설정: 모델 학습 전에 학습에 대한 설정을 수행한다. / 손실 함수 및 최적화 방법을 정의.
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= ["accuracy"])

## loss function: 훈련하는 동안 모델의 오차를 측정. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 한다.
## optimizer: 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정.
## metrics: 훈련 단계와 테스트 단계를 모니터링하기 위해 사용.

train_dg = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=30,
                                                            width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            # brightness_range=[0.5, 1.5],
                                                            # shear_range=0.1,
                                                            zoom_range=0.1,
                                                            fill_mode='constant',
                                                            cval=0.0, # fill_mode = "constant"일 때 씀
                                                            horizontal_flip=True,
                                                            # vertical_flip=True,
                                                            )

MODEL_SAVE_FOLDER_PATH = 'ResNet50_none_resize_augO/model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

checkpoint = ModelCheckpoint(filepath=f'{MODEL_SAVE_FOLDER_PATH}checkpoint_{model.name}.h5', monitor='val_loss', verbose=1, save_best_only=True)

# 모델 학습: training set을 이용해서 구성한 모델로 학습시킴. / fit(입력데이터, 결과(label값)데이터, 한 번에 학습할 때 사용하는 데이터 개수, 학습 데이터 반복 횟수)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")


# history = model.fit(x_train_resized, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, reduce_lr, early_stopping])
history = model.fit_generator(train_dg.flow(x_train_resized, y_train, batch_size=batch_size),
                              steps_per_epoch=math.ceil(y_train.shape[0]/batch_size),
                              epochs=epochs,
                              validation_data=(x_test_resized, y_test),
                              callbacks=[checkpoint, reduce_lr, early_stopping])

# # test set을 validation set으로 써도 되는건지 모르겠음
#
# history = model.load_weights(f'{MODEL_SAVE_FOLDER_PATH}checkpoint_{model.name}.h5')


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

# evaluate the trained model
# 모델 평가: 준비된 test set으로 학습한 모델을 평가한다.
score = model.evaluate(x_test_resized, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


## predict
predicted_result = model.predict(x_test_resized)
predicted_labels = np.argmax(predicted_result, axis=1)
# 이미지 예측 파일 분류
path_list = []

for i in range(num_classes):
    path_list.append('./cifar10/ResNet50_none_resize_augO/classified_img/' + str(i))

for path in path_list:
  if not os.path.exists(path):
      os.makedirs(path)

for n in range(len(predicted_labels)):
    cv2.imwrite(path_list[predicted_labels[n]]+ '/' + str(n) + '.jpg', x_test_resized[n]*255)

y_test = np.argmax(y_test, axis=1)

confusion_matrix = confusion_matrix(y_test, predicted_labels)
print("\nconfusion matrix")
print(confusion_matrix)


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


dir_url = 'ResNet50_none_resize_augO/classified_img/'

files = os.listdir(dir_url)

for i in range(len(y_test)):
    if predicted_labels[i] != y_test[i]:
        temp_url = dir_url + files[predicted_labels[i]]
        os.rename(temp_url + '/' + str(i) + '.jpg', temp_url + '/' + str(i) + '_' + str(y_test[i]) + '.jpg')
# ResNet50_none_resize_augX 저장 안됨







# ResNet50_none 잘못 저장돼있음