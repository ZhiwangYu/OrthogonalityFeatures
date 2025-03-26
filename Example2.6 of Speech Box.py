# -*- coding: utf-8 -*-
"""
主程序，通过更改路径来选择不同数据集的数据，包含是否筛选，音频的归一化（可能为KF+NOL等卷积核在前面表现不佳的原因），
以及加噪，同时这里的卷积核变更到直接积分的形式。
"""

import os
import pathlib
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D, Activation, Layer
import math 
from scipy.integrate import tplquad,dblquad,quad
from scipy.linalg import expm

DATASET_PATH = 'data/speech_commands' # 修改为您的数据路径    'data/speech_box/audio_segment_no012_selected'
                                        #        'F:/Datasets/TIMIT_all' 'data/LJSpeech-1.1/audio_segment_all_0.5_no012'

data_dir = pathlib.Path(DATASET_PATH)
all_class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])

# 存储每个类别对应的音频文件路径
class_audio_files = {}

for class_name in all_class_names:
    class_dir = data_dir / class_name
    # 获取该类别下所有的音频文件，包括子文件夹中的文件
    audio_files = [str(p.resolve()) for p in class_dir.glob('**/*.wav')]  # 请根据您的文件格式调整
    class_audio_files[class_name] = audio_files

# 筛选类别和随机选择文件
selected_class_files = {}       #调控这里来实现是否筛选
min_samples_per_class = 0      
max_samples_per_class = 500000000

for class_name, audio_files in class_audio_files.items():
    num_files = len(audio_files)
    if num_files >= min_samples_per_class:
        selected_files = random.sample(audio_files, min(max_samples_per_class, num_files))
        selected_class_files[class_name] = selected_files
    else:
        # 舍弃该类别
        print(f"类别 {class_name} 的样本数量不足 {min_samples_per_class}，已舍弃（样本数：{num_files}）。")

# 构建文件路径和标签列表
all_file_paths = []
all_labels = []
label_names = sorted(selected_class_files.keys())
label_to_index = {name: index for index, name in enumerate(label_names)}

for class_name, file_paths in selected_class_files.items():
    all_file_paths.extend(file_paths)
    all_labels.extend([label_to_index[class_name]] * len(file_paths))

# 将文件路径和标签转换为 TensorFlow 字符串张量
all_file_paths = tf.convert_to_tensor(all_file_paths)
all_labels = tf.convert_to_tensor(all_labels)

# 创建 Dataset
dataset = tf.data.Dataset.from_tensor_slices((all_file_paths, all_labels))

# 通过 shuffle 方法实现随机性
dataset = dataset.shuffle(buffer_size=len(all_file_paths), seed=0)

# 定义音频加载和预处理函数
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    return audio

def process_path(file_path, label):
    audio = decode_audio(file_path)
    # 确保音频长度为16000
    audio = audio[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    return audio, label

AUTOTUNE = tf.data.AUTOTUNE

dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

# 划分训练和验证集
dataset_size = len(all_file_paths)
print(f"总样本数量：{dataset_size}")

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

batch_size = 64
train_ds = train_ds.shuffle(1000).batch(batch_size).cache().prefetch(AUTOTUNE)
val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

# Define a function to add Gaussian white noise with specified SNR and normalize
def add_noise_with_snr(audio, labels, desired_snr_db=None):
    if desired_snr_db is None:
        return audio, labels
    rms_signal = tf.sqrt(tf.reduce_mean(tf.square(audio)))
    snr_linear = tf.math.pow(10.0, desired_snr_db / 20.0)
    rms_noise = rms_signal / snr_linear
    noise = tf.random.normal(tf.shape(audio))
    noise = noise * (rms_noise / tf.sqrt(tf.reduce_mean(tf.square(noise))))
    audio_noisy = audio + noise
    max_amplitude = tf.reduce_max(tf.abs(audio_noisy))
    audio_noisy = tf.cond(tf.greater(max_amplitude, 1.0), lambda: audio_noisy / max_amplitude, lambda: audio_noisy)
    return audio_noisy, labels

# Add Gaussian white noise to the training dataset
train_ds = train_ds.map(lambda audio, labels: add_noise_with_snr(audio, labels, desired_snr_db=None), 
                        num_parallel_calls=tf.data.AUTOTUNE)

# 定义 Spectrogram 转换
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)

# 获取输入形状和标签数量
for spectrograms, labels in train_spectrogram_ds.take(1):
    input_shape = spectrograms.shape[1:]
    break

num_labels = len(label_names)

# 定义归一化层
norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(lambda spec, label: spec))

# 设定种子（可选）
np.random.seed(42)

# 1. 生成16个随机的反对称矩阵
num_kernels = 16
anti_sym_matrices = []
for _ in range(num_kernels):
    M = np.random.randn(3, 3) * 2  # 增加方差范围
    A = M - M.T  # 生成反对称矩阵
    anti_sym_matrices.append(A)

# 2. 计算每个反对称矩阵的指数映射
rot_matrices = []
for A in anti_sym_matrices:
    R = expm(A)
    rot_matrices.append(R)

# 3. 将旋转矩阵应用于原始矩阵
A_matrices = [
    np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]),
    np.array([[1, 0, 1],
              [1, 0, 1],
              [1, 0, 1]])
    #np.array([[1, -2, 1],
    #          [1, -2, 1],
    #          [1, -2, 1]])/np.sqrt(3)
]
A_matrices += [-A for A in A_matrices]

transformed_matrices = []
for R in rot_matrices:
    for A in A_matrices:
        A_new = R @ A
        transformed_matrices.append(A_new)
        #transformed_matrices.append(A_new.T)

# 4. 转换为 NumPy 数组
transformed_matrices = np.array(transformed_matrices)

# 定义您的模型，包括 SphereFeatures 层
class SphereFeatures(tf.keras.layers.Layer):
    def __init__(self, transformed_matrices, output_dim, kernel_size=(3, 3), strides=(1, 1), **kwargs):
        super(SphereFeatures, self).__init__(**kwargs)
        self.transformed_matrices = transformed_matrices  # 保存 transformed_matrices
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        # 确保 transformed_matrices 是 numpy 数组
        if isinstance(self.transformed_matrices, list):
            self.transformed_matrices = np.array(self.transformed_matrices)

        # 调整 transformed_matrices 的形状
        kernels = self.transformed_matrices.reshape(-1, self.kernel_size[0], self.kernel_size[1])
        kernels = kernels[:self.output_dim]  # 如果数量过多，可以截取到 output_dim 个

        # 初始化卷积核为不可训练的常量
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = np.zeros(shape=kernel_shape)
        for i in range(self.output_dim):
            for c in range(input_shape[-1]):
                temp[:, :, c, i] = kernels[i]

        # 将卷积核设置为不可训练的常量张量
        self.kernel = tf.constant(temp, dtype=tf.float32)

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=[1, *self.strides, 1], padding='SAME')
        return output

# 请确保 transformed_matrices 已经定义
# 如果没有，请按照您之前的代码生成 transformed_matrices

class CircleFeatures(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CircleFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        # Generate theta values uniformly from 0 to 2π
        theta_list = np.linspace(0, 2 * np.pi, self.output_dim, endpoint=False)
        
        # Initialize the kernel
        kernel_shape = (3, 3, input_channels, self.output_dim)
        initial_kernel = np.zeros(kernel_shape, dtype=np.float32)
        
        # Compute the kernel values
        theta_2=np.pi/2
        for i in range(input_channels):
            for j in range(self.output_dim):
                theta_1 = theta_list[j]
                for n in range(3):
                    for m in range(3):
                        # Compute kernel using your Filter function logic
                        kernel_value = self.compute_filter(n, m, theta_1, theta_2)
                        initial_kernel[n, m, i, j] = kernel_value
                        
        # Set the kernel as a constant tensor to make it untrainable
        self.kernel = tf.constant(initial_kernel, dtype=tf.float32)
        
    def call(self, inputs):
        # Perform convolution with the specified kernel
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return output
    
    def compute_filter(self, n, m, theta_1, theta_2):
        # Define the integration limits
        x_lower = -1 + 2 * m / 3
        x_upper = -1 + 2 * (m + 1) / 3
        y_lower = -1 + 2 * n / 3
        y_upper = -1 + 2 * (n + 1) / 3
        
        # Perform the double integral
        result, _ = dblquad(
            lambda y, x: self.F_K(theta_1, theta_2, x, y),
            x_lower, x_upper,
            lambda x: y_lower,
            lambda x: y_upper
        )
        return result
    
    def F_K(self, theta_1, theta_2, x, y):
        def Q(t):
            return 2 * t**2 - 1
        
        t = np.cos(theta_1) * x + np.sin(theta_1) * y
        value = np.sin(theta_2) * t + np.cos(theta_2) * Q(t)
        return value


class KleinFeatures(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(KleinFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # 计算输出维度的平方根，确保是整数
        sqrt_output_dim = int(np.sqrt(self.output_dim))
        assert sqrt_output_dim ** 2 == self.output_dim, "输出维度必须是完全平方数"

        # 生成角度列表，数量为 sqrt_output_dim
        theta_1_list = np.linspace(0, np.pi, sqrt_output_dim, endpoint=False)  # θ₁ 范围：[0, π)
        theta_2_list = np.linspace(0, 2 * np.pi, sqrt_output_dim, endpoint=False)  # θ₂ 范围：[0, 2π)

        # 初始化卷积核
        kernel_shape = (3, 3, input_channels, self.output_dim)
        initial_kernel = np.zeros(kernel_shape, dtype=np.float32)

        # 按角度组合计算卷积核的值
        idx = 0  # 输出通道索引
        for theta_1 in theta_1_list:
            for theta_2 in theta_2_list:
                for i in range(input_channels):
                    for n in range(3):
                        for m in range(3):
                            # 计算卷积核值
                            kernel_value = self.compute_filter(n, m, theta_1, theta_2)
                            initial_kernel[n, m, i, idx] = kernel_value
                idx += 1
                if idx >= self.output_dim:
                    break
            if idx >= self.output_dim:
                break

        # 将卷积核设置为不可训练
        self.kernel = tf.constant(initial_kernel, dtype=tf.float32)

    def call(self, inputs):
        # 使用指定的卷积核进行卷积操作
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return output

    def compute_filter(self, n, m, theta_1, theta_2):
        # 定义积分的边界
        x_lower = -1 + 2 * m / 3
        x_upper = -1 + 2 * (m + 1) / 3
        y_lower = -1 + 2 * n / 3
        y_upper = -1 + 2 * (n + 1) / 3

        # 进行二重积分
        result, _ = dblquad(
            lambda y, x: self.F_K(theta_1, theta_2, x, y),
            x_lower, x_upper,
            lambda x: y_lower,
            lambda x: y_upper
        )
        return result

    def F_K(self, theta_1, theta_2, x, y):
        def Q(t):
            return 2 * t ** 2 - 1

        # 根据公式计算 F_K
        t = np.cos(theta_1) * x + np.sin(theta_1) * y
        value = np.sin(theta_2) * t + np.cos(theta_2) * Q(t)
        return value

    
class CircleOneLayer(Conv2D):
    def __init__(self, distance=5/16, *args, **kwargs):
        super(CircleOneLayer, self).__init__(*args, **kwargs)
        self.distance = distance 

    def build(self, input_shape):
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters)
        temp_3 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        shape_temp = (self.kernel_size[0], self.kernel_size[1])
        for i in range(input_shape[-1]):
            for j in range(self.filters): 
                if abs(i/input_shape[-1]-j/self.filters) < self.distance or abs(i/input_shape[-1]-j/self.filters) > 1-self.distance:
                    temp_3[:,:,i,j] = tf.ones(shape=shape_temp, dtype=tf.float32).numpy()
        self.mask = tf.constant(temp_3)
        self.kernel = self.add_weight(name='kernel', shape=kernel_shape, initializer='glorot_uniform',trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)

    def call(self, inputs):
        # Apply the mask to the kernel
        self.kernel.assign(self.kernel * self.mask)
        return super(CircleOneLayer, self).call(inputs)
    
class KleinOneLayer(Conv2D):
    def __init__(self, distance=2, *args, **kwargs):
        super(KleinOneLayer, self).__init__(*args, **kwargs)
        self.distance = distance

    def build(self, input_shape):
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters)
        
        def Q(t):
            return 2*t**2-1
        
        def F_K(theta_1, theta_2, x, y):
            return math.sin(theta_2)*(math.cos(theta_1)*x+math.sin(theta_1)*y)+math.cos(theta_2)*Q(math.cos(theta_1)*x+math.sin(theta_1)*y)
        
        def D(a_1,a_2,b_1,b_2): 
            return np.power(dblquad(lambda y,x:(F_K(a_1, a_2, x, y)-F_K(b_1, b_2, x, y))**2,#函数
                      -1,#x下界0
                      1,#x上界pi
                      lambda x:-1,#y下界x^2
                      lambda x:1), 1/2)[0]#y上界2*x
        
        temp_3 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            for j in range(self.filters):
                if D(i%int(math.sqrt(input_shape[-1])) * math.pi/int(math.sqrt(input_shape[-1])), i//int(math.sqrt(input_shape[-1])) *2* math.pi/int(math.sqrt(input_shape[-1])), j%int(math.sqrt(self.filters)) * math.pi/int(math.sqrt(self.filters)), j//int(math.sqrt(self.filters))*2*math.pi/int(math.sqrt(self.filters)))<=self.distance:  
                    temp_3[:,:,i,j] = tf.ones(shape=(self.kernel_size[0], self.kernel_size[1]), dtype=tf.float32).numpy()
        self.mask = tf.constant(temp_3)
        self.kernel = self.add_weight(name='kernel', shape=kernel_shape, initializer='glorot_uniform',trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        
        def call(self, inputs):
            # Apply the mask to the kernel
            self.kernel.assign(self.kernel * self.mask)
            return super(KleinOneLayer, self).call(inputs)

#CNN
model1 = models.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    #layers.Conv2D(128, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_labels), 
    #layers.Dense(1, activation='sigmoid'),
])

model1.summary()

model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #loss='binary_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 20
history1 = model1.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)


# 构建模型
model2 = models.Sequential([
    layers.Input(shape=input_shape),
    norm_layer,
    SphereFeatures(transformed_matrices=transformed_matrices, output_dim=64),
    layers.Activation('relu'), 
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_labels), 
])

model2.summary()

# 编译模型
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# 训练模型
EPOCHS = 20
history2 = model2.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

# 构建模型
model3 = models.Sequential([
    layers.Input(shape=input_shape),
    norm_layer,
    CircleFeatures(output_dim=64),
    layers.Activation('relu'), 
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_labels), 
])

model3.summary()

# 编译模型
model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# 训练模型
EPOCHS = 20
history3 = model3.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

# 构建模型
model4 = models.Sequential([
    layers.Input(shape=input_shape),
    norm_layer,
    KleinFeatures(output_dim=64),
    layers.Activation('relu'), 
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_labels), 
])

model4.summary()

# 编译模型
model4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# 训练模型
EPOCHS = 20
history4 = model4.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history1.history['loss'], label = 'Normal', color = 'blue')
plt.plot(history2.history['loss'], label = 'OF+NOL', color = 'purple')
plt.plot(history3.history['loss'], label = 'CF+NOL', color = 'red')
plt.plot(history4.history['loss'], label = 'KF+NOL', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='lower right')


plt.subplot(1,2,2)
plt.plot(history1.history['val_accuracy'], label = 'Normal', color = 'blue')
plt.plot(history2.history['val_accuracy'], label = 'OF+NOL', color = 'purple')
plt.plot(history3.history['val_accuracy'], label = 'CF+NOL', color = 'red')
plt.plot(history4.history['val_accuracy'], label = 'KF+NOL', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('val_Acc')
plt.legend(loc='lower right')
plt.show()