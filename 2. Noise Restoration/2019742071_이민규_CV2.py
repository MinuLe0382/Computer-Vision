import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model

"""
* 개발환경 : Anaconda & VScode
conda version : 23.7.4
conda-build version : 3.26.1
python version : 3.11.5.final.0
"""

# salt and pepper noise를 추가하는 함수 정의
def snp_noise_maker_v3_optimized(gray_img, noise_ratio=0.2):
    noisy_img = np.copy(gray_img)
    mask = np.random.rand(*gray_img.shape) < noise_ratio
    # 비율에 따라 랜덤으로 노이즈를 생성할 좌표 list를 생성하여 이를 mask로 활용
    noise = np.random.choice([0, 255], size=gray_img.shape, p=[0.5, 0.5])
    # salt와 pepper의 비율은 1:1로 설정
    noisy_img[mask] = noise[mask]
    return noisy_img

#가우시안 노이즈를 추가하는 함수 정의
def Gaussian_noise_maker_optimized(gray_img, sigma=10):
    noise = np.random.normal(0, sigma, gray_img.shape)
    # numpy 내장함수 random.normal은 가우시안 노이즈와 동일한 역할 수행
    imgdata_Gaussian_noise = gray_img + noise
    return imgdata_Gaussian_noise

# alpha trimmed mean filter를 적용하는 함수 정의
def alpha_trimmed_mean_filter(image, filter_size, alpha):
    # window에서 몇개의 픽셀을 제거할 것인가?
    trim_amount = int((filter_size * filter_size) * alpha)

    # 사용하는 필터의 크기에 따라 zero padding을 적용
    pad_size = filter_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    filtered_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            # 각 픽셀을 중심으로 필터사이즈 만큼의 이웃픽셀을 추출하여 1-dimension으로 변경.
            window = padded_image[i:i + filter_size, j:j + filter_size].flatten()
            # 뽑아낸 배열을 크기순서대로 정렬
            window_sorted = np.sort(window)
            # 좌우 극단값을 제거하고 남은 평균을 지정한 픽셀의 값으로 결정
            trimmed_window = window_sorted[trim_amount:-trim_amount]
            filtered_image[i, j] = np.mean(trimmed_window)

    return filtered_image

# PSNR을 계산하는 함수 정의
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 ) #MSE 구하는 코드

    #MSE가 0이면 100으로 정의
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 이미지 배열의 PSNR의 평균과 분산을 계산하는 함수 정의
def calc_psnr(original_images, restored_images):
    psnr_values = []

    for i in range(len(original_images)):
        psnr_value = psnr(original_images[i], restored_images[i])
        psnr_values.append(psnr_value)

    average_psnr = np.mean(psnr_values)
    variance_psnr = np.var(psnr_values)
    return average_psnr, variance_psnr

# 기계학습에 사용하기전 데이터를 전처리(0~1 사이의 값으로 정규화)
def preprocess(array):
    pre_array = array.astype("float32") / 255.0
    pre_array = np.reshape(pre_array, (len(array), 256, 256, 1))
    return pre_array

#정규화를 복원
def deprocess(pre_array):
    de_array = np.reshape(pre_array, (len(pre_array), 256, 256))
    de_array = de_array * 255.0
    de_array = de_array.astype("uint8")
    
    return de_array

# 데이터 다운로드
dataset_resize = np.load('dataset.npy')
train_data = dataset_resize[:3000]
test_data = dataset_resize[3000:]
data_shape = train_data[0].shape

train_data_snp_noisy_images = []
train_data_gaussian_noisy_images = []
test_data_snp_noisy_images = []
test_data_gaussian_noisy_images = []


# 이미지에 노이즈 적용
for img in train_data:
    train_data_snp_noisy_images.append(snp_noise_maker_v3_optimized(img))

for img in train_data:
    train_data_gaussian_noisy_images.append(Gaussian_noise_maker_optimized(img))

for img in test_data[:100]:
    test_data_snp_noisy_images.append(snp_noise_maker_v3_optimized(img))

for img in test_data[:100]:
    test_data_gaussian_noisy_images.append(Gaussian_noise_maker_optimized(img))

train_data_snp_noisy_images = np.array(train_data_snp_noisy_images)
train_data_gaussian_noisy_images = np.array(train_data_gaussian_noisy_images)
test_data_snp_noisy_images = np.array(test_data_snp_noisy_images)
test_data_gaussian_noisy_images = np.array(test_data_gaussian_noisy_images)
# Train data, Test data에 각각 Salt and Pepper noise와 가우시안 noise를 적용


# 노이즈가 적용된 이미지에 Alpha Trimmed Mean Filter 적용
snp_3x3_alpha02_images = [] # snp 노이즈에 3 * 3, alpha 0.2 filter 적용
snp_5x5_alpha04_images = [] # snp 노이즈에 5 * 5, alpha 0.4 filter 적용
gaussian_3x3_alpha02_images = [] # 가우시안 노이즈에 3 * 3, alpha 0.2 filter 적용
gaussian_5x5_alpha04_images = [] # 가우시안 노이즈에 5 * 5, alpha 0.4 filter 적용

for img in test_data_snp_noisy_images:
    snp_3x3_alpha02_images.append(alpha_trimmed_mean_filter(img, 3, 0.2))
    snp_5x5_alpha04_images.append(alpha_trimmed_mean_filter(img, 5, 0.4))

for img in test_data_gaussian_noisy_images:
    gaussian_3x3_alpha02_images.append(alpha_trimmed_mean_filter(img, 3, 0.2))
    gaussian_5x5_alpha04_images.append(alpha_trimmed_mean_filter(img, 5, 0.4))

snp_3x3_alpha02_images = np.array(snp_3x3_alpha02_images)
snp_5x5_alpha04_images = np.array(snp_5x5_alpha04_images)
gaussian_3x3_alpha02_images = np.array(gaussian_3x3_alpha02_images)
gaussian_5x5_alpha04_images = np.array(gaussian_5x5_alpha04_images)


# 학습에 사용할 train data, test data, 노이즈 이미지를 정규화
normal_train_data = preprocess(train_data)
normal_test_data = preprocess(test_data)
normal_train_data_snp_noisy_images = preprocess(train_data_snp_noisy_images)
normal_train_data_gaussian_noisy_images = preprocess(train_data_gaussian_noisy_images)
normal_test_data_snp_noisy_images = preprocess(test_data_snp_noisy_images)
normal_test_data_gaussian_noisy_images = preprocess(test_data_gaussian_noisy_images)


# 모델 1. Salt and Pepper 복원용 32채널 오토인코더
input = layers.Input(shape=(256, 256, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder_32c = Model(input, x)
autoencoder_32c.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder_32c.summary()

# 모델 2. Gaussian noise 복원용 32채널 오토인코더
input = layers.Input(shape=(256, 256, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder_32c_gauss = Model(input, x)
autoencoder_32c_gauss.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder_32c_gauss.summary()

#모델 3. Salt and Pepper 복원용 64채널 오토인코더
input = layers.Input(shape=(256, 256, 1))

# Encoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder_64c = Model(input, x)
autoencoder_64c.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder_64c.summary()

# 모델 4. Gaussian noise 복원용 64채널 오토인코더
input = layers.Input(shape=(256, 256, 1))

# Encoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder_64c_gauss = Model(input, x)
autoencoder_64c_gauss.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder_64c_gauss.summary()


# 모델 학습, 배치크기 32, 에포크 10
autoencoder_32c.fit(normal_train_data_snp_noisy_images, normal_train_data, batch_size = 32, epochs = 10)
autoencoder_64c.fit(normal_train_data_snp_noisy_images, normal_train_data, batch_size = 32, epochs = 10)
autoencoder_32c_gauss.fit(normal_train_data_gaussian_noisy_images, normal_train_data, batch_size = 32, epochs = 10)
autoencoder_64c_gauss.fit(normal_train_data_gaussian_noisy_images, normal_train_data, batch_size = 32, epochs = 10)

# test 데이터로 예측수행
prediction32 = autoencoder_32c.predict(normal_test_data_snp_noisy_images)
prediction64 = autoencoder_64c.predict(normal_test_data_snp_noisy_images)
prediction32_gauss = autoencoder_32c_gauss.predict(normal_test_data_gaussian_noisy_images)
prediction64_gauss = autoencoder_64c_gauss.predict(normal_test_data_gaussian_noisy_images)


# 각 복원 결과별 PSNR의 평균과 분산 출력
print(calc_psnr(test_data, snp_3x3_alpha02_images))
print(calc_psnr(test_data, snp_5x5_alpha04_images))
print(calc_psnr(test_data, gaussian_3x3_alpha02_images))
print(calc_psnr(test_data, gaussian_5x5_alpha04_images))

print(calc_psnr(test_data, deprocess(prediction32)))
print(calc_psnr(test_data, deprocess(prediction64)))
print(calc_psnr(test_data, deprocess(prediction32_gauss)))
print(calc_psnr(test_data, deprocess(prediction64_gauss)))
# 노이즈의 PSNR 계산
print(calc_psnr(test_data, test_data_snp_noisy_images))
print(calc_psnr(test_data, test_data_gaussian_noisy_images))


# 복원결과출력, 비교 (Salt and Pepper Noise)
plt.imshow(test_data[0], cmap = 'gray')
plt.title('Original Image')
plt.show()
plt.imshow(test_data_snp_noisy_images[0], cmap = 'gray')
plt.title('Salt and Pepper noise Image')
plt.show()
plt.imshow(snp_3x3_alpha02_images[0], cmap = 'gray')
plt.title('SnP noise restored, 3 * 3 filter')
plt.show()
plt.imshow(snp_5x5_alpha04_images[0], cmap = 'gray')
plt.title('SnP noise restored, 5 * 5 filter')
plt.show()
plt.imshow(prediction32[0], cmap = 'gray')
plt.title('SnP noise restored, autoencoder 32-channel ')
plt.show()
plt.imshow(prediction64[0], cmap = 'gray')
plt.title('SnP noise restored, autoencoder 64-channel')
plt.show()

# 복원결과출력, 비교 (Gaussian Noise)
plt.imshow(test_data[0], cmap = 'gray')
plt.title('Original Image')
plt.show()
plt.imshow(test_data_gaussian_noisy_images[0], cmap = 'gray')
plt.title('Gaussian noise Image')
plt.show()
plt.imshow(gaussian_3x3_alpha02_images[0], cmap = 'gray')
plt.title('Gaussian noise restored, 3 * 3 filter')
plt.show()
plt.imshow(gaussian_5x5_alpha04_images[0], cmap = 'gray')
plt.title('Gaussian noise restored, 5 * 5 filter')
plt.show()
plt.imshow(prediction32_gauss[0], cmap = 'gray')
plt.title('Gaussian noise restored, autoencoder 32-channel ')
plt.show()
plt.imshow(prediction64_gauss[0], cmap = 'gray')
plt.title('Gaussian noise restored, autoencoder 64-channel')
plt.show()