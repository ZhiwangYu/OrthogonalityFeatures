# -*- coding: utf-8 -*-
"""
此程序初步验证了OF+NOL的可行性，主要输出为其与KF+NOL,CF+NOL,以及NOL+NOL的对比。
注意：本实验已废弃，如果读者感兴趣，可以考虑和‘9.main.py’程序对比（此程序缺少归一化这一步骤）。
"""

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import math
import librosa

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from scipy.integrate import tplquad,dblquad,quad


DATASET_PATH =  'data/speech_box/audio_segment_no012_selected' #'data/speech_commands'

data_dir = pathlib.Path(DATASET_PATH)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

  
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
num_labels = len(label_names)

norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(lambda spec, label: spec))

# 定义原始矩阵
A_matrices = [
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
]
# 添加它们的负矩阵
A_matrices += [-A for A in A_matrices]

# 定义角度
alpha_values = [0, np.pi/2, np.pi, 3*np.pi/2]  # 4 个值
beta_values = [0, np.pi]                       # 2 个值
gamma_values = [0, np.pi]                      # 2 个值

# 存储变换后的矩阵
transformed_matrices = []

# 生成正交矩阵并进行变换
for alpha in alpha_values:
    for beta in beta_values:
        for gamma in gamma_values:
            # 构造旋转矩阵
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]
            ])
            R_y = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])
            R_z = np.array([
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]
            ])
            R = R_z @ R_y @ R_x  # 总体旋转矩阵

            # 对每个原始矩阵进行变换
            for A in A_matrices:
                A_new = R @ A  # 左乘旋转矩阵
                transformed_matrices.append(A_new)
                


# 转换为 TensorFlow 张量
transformed_matrices = np.array(transformed_matrices)
transformed_tensors = tf.constant(transformed_matrices, dtype=tf.float32)

class SphereFeatures(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1), **kwargs):
        super(SphereFeatures, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        # 假设输入是四维张量：[batch, height, width, channels]
        
        # 构造你的变换矩阵 transformed_matrices
        # 这里假设 transformed_matrices 已经按照需求生成
        
        # 确保 transformed_matrices 的形状正确
        kernels = transformed_matrices.reshape(-1, self.kernel_size[0], self.kernel_size[1])
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
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=[1, *self.strides, 1], padding='VALID')
        return output



class TorusFeatures(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(TorusFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        k1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        k1 = k1/np.sqrt(6)
        k2 = k1.T
        k3 = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
        k3 = k3/np.sqrt(18)
        k4 = k3.T
        v2 = np.array([[-2, 0, 2], [-1, 0, 1], [0, 0, 0]])
        v2 = v2/np.sqrt(10)
        v21 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        v21 = v2/np.sqrt(3)
        v22 = np.array([[-3, 0, 3], [-2, 0, 2], [-1, 0, 1]])
        v22 = v22/np.sqrt(28)
        thetas = np.arange(0, 2*np.pi, np.pi/4)
        v = []
        for theta_1 in thetas:
            for theta_2 in thetas:
                #甜甜圈在v2方向压成一点
                w = 25/49*(np.cos(theta_1/2)*k2+np.sin(theta_1/2)*v22)+24/49*np.cos(theta_1/2)**2*(np.cos(theta_2)*(np.cos(theta_1/2)*k2+np.sin(theta_1/2)*v22)  #v2,v22可互换
                                            +np.sin(theta_2)*k4) #k4,v21可互换
                v.append(w/3)
        v = np.array(v)
        
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = tf.zeros(shape=kernel_shape).numpy()
        for i in range(input_shape[-1]):
            temp[:,:,i,:] = v.reshape(self.kernel_size[0], self.kernel_size[1], self.output_dim)
        self.kernel = tf.constant(temp)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='VALID')
        return output
    
class Torus1Features(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(Torus1Features, self).__init__(**kwargs)

    def build(self, input_shape):
        k1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        k1 = k1/np.sqrt(6)
        k2 = k1.T
        k3 = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
        k3 = k3/np.sqrt(18)
        k4 = k3.T
        v2 = np.array([[-2, 0, 2], [-1, 0, 1], [0, 0, 0]])
        v2 = v2/np.sqrt(10)
        v21 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        v21 = v21/np.sqrt(3)
        v22 = np.array([[-3, 0, 3], [-2, 0, 2], [-1, 0, 1]])
        v22 = v22/np.sqrt(28)
        #半个环面，v2对应位置缩成一点，同时v1附近点密集
        
        thetas = np.arange(np.pi/8, np.pi, np.pi/8)
        v = []
        
        for theta_1 in thetas:
            theta_2 = 0
            while theta_2<2*np.pi:
                if int(abs(theta_1/np.pi*8-4))==0:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/8
                if int(abs(theta_1/np.pi*8-4))==1:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*v22)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/6
                if int(abs(theta_1/np.pi*8-4))==2:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/4
                if int(abs(theta_1/np.pi*8-4))==3:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/2
        v = np.array(v)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = tf.zeros(shape=kernel_shape).numpy()
        for i in range(input_shape[-1]):
            temp[:,:,i,:] = v.reshape(self.kernel_size[0], self.kernel_size[1], self.output_dim)
        self.kernel = tf.constant(temp)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='VALID')
        return output

#CNN
model1 = models.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
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

EPOCHS = 5
history1 = model1.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

y1_pred = model1.predict(test_spectrogram_ds)

#y1_pred = tf.where(y1_pred > 0.5, 1, 0) # Binary classification threshold 
#y1_pred = tf.squeeze(y1_pred)
y1_pred = tf.argmax(y1_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx1 = tf.math.confusion_matrix(y_true, y1_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx1,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Sphere
model2 = models.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    SphereFeatures(64), 
    layers.Activation('relu'), 
    #layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_labels), 
    #layers.Dense(1, activation='sigmoid'),
])

model2.summary()

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #loss='binary_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 5
history2 = model2.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

y2_pred = model2.predict(test_spectrogram_ds)

#y2_pred = tf.where(y2_pred > 0.5, 1, 0) # Binary classification threshold 
#y2_pred = tf.squeeze(y2_pred)
y2_pred = tf.argmax(y2_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx2 = tf.math.confusion_matrix(y_true, y2_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx2,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Torus
model3 = models.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    TorusFeatures(64), 
    layers.Activation('relu'), 
    #layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    #layers.Conv2D(256, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_labels), 
    #layers.Dense(1, activation='sigmoid'),
])

model3.summary()

model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #loss='binary_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 5
history3 = model3.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

y3_pred = model3.predict(test_spectrogram_ds)

#y3_pred = tf.where(y3_pred > 0.5, 1, 0) # Binary classification threshold 
#y3_pred = tf.squeeze(y3_pred)
y3_pred = tf.argmax(y3_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx3 = tf.math.confusion_matrix(y_true, y3_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx3,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Weighted Torus
model4 = models.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    Torus1Features(64), 
    layers.Activation('relu'), 
    #layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    #layers.Conv2D(256, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_labels), 
    #layers.Dense(1, activation='sigmoid'),
])

model4.summary()

model4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #loss='binary_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 5
history4 = model4.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

y4_pred = model4.predict(test_spectrogram_ds)

#y4_pred = tf.where(y4_pred > 0.5, 1, 0) # Binary classification threshold 
#y4_pred = tf.squeeze(y4_pred)
y4_pred = tf.argmax(y4_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx4 = tf.math.confusion_matrix(y_true, y4_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx4,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(history1.history['loss'], label = 'Normal', color = 'blue')
plt.plot(history2.history['loss'], label = 'Sphere', color = 'purple')
plt.plot(history3.history['loss'], label = 'Torus', color = 'red')
plt.plot(history4.history['loss'], label = 'W-Torus', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='lower right')


plt.subplot(1,2,2)
plt.plot(history1.history['val_accuracy'], label = 'Normal', color = 'blue')
plt.plot(history2.history['val_accuracy'], label = 'Sphere', color = 'purple')
plt.plot(history3.history['val_accuracy'], label = 'Torus', color = 'red')
plt.plot(history4.history['val_accuracy'], label = 'W-Torus', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('val_Acc')
plt.legend(loc='lower right')
plt.show()

