# -*- coding: utf-8 -*-
"""
此段程序主要提取卷积神经网络的的权向量。其中包含的卷积核定义是根据PCA的结果合理猜测得到（虽然实验效果很差），
并用在接下来的程序会体现
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


DATASET_PATH = 'data/LJSpeech-1.1/audio_segment_all_0.5_no012'

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

weights_list_1=[]
weights_list_2=[]
weights_list_3=[]

class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch_count + 1) % 5 == 0:
            # 在每五个 epoch 结束后获取模型的权重并保存在列表中
            #weights_first_conv_layer = model.get_layer('conv1').get_weights()
            weights1 = model.layers[1].get_weights()
            weights1 = weights1[0] 
            weights_list_1.append(weights1)
            weights2 = model.layers[3].get_weights()
            weights2 = weights2[0] 
            weights_list_2.append(weights2)
            weights3 = model.layers[5].get_weights()
            weights3 = weights3[0] 
            weights_list_3.append(weights3)
        self.epoch_count += 1

norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(lambda spec, label: spec))

class ConstantFeatures(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(ConstantFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        v = np.load('vectors_sin_640.npy')
        indices = np.random.choice(v.shape[0], 64, replace=False)
        v_sampled = v[indices].reshape((64, 3, 3))
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = tf.zeros(shape=kernel_shape).numpy()
        for i in range(input_shape[-1]):
            temp[:,:,i,:] = v_sampled.reshape(self.kernel_size[0], self.kernel_size[1], self.output_dim)
        self.kernel = tf.constant(temp)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='VALID')
        return output

class CircleFeatures(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(CircleFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        k1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        k1 = k1/np.sqrt(6)
        k2 = k1.T
        k3 = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
        k3 = k3/np.sqrt(18)
        k4 = k3.T
        v2 = np.array([[-2, 0, 2], [-1, 0, 1], [0, 0, 0]])
        v2 = v2/np.sqrt(10)
        thetas = np.arange(0, np.pi, np.pi/60)
        v = []
        #for theta_1 in thetas:
        for theta_2 in thetas:
            """
            k5 = np.array([[-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                           [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                           [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)]])
            k5 = k5/np.linalg.norm(k5)
            k6 = k5.T   #k1,k2,k3,k4的某种组合
            w = k6  #np.cos(theta_1)*k5+np.sin(theta_1)*
            """
            w = np.cos(theta_2)*k2+np.sin(theta_2)*k4
            
            v.append(w/4)
        
        #第三向量顶点
        #k0 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        selected_thetas = thetas[1::15]
        selected_thetas = [2*theta for theta in selected_thetas]
        for theta_1 in selected_thetas:
            k5 = np.array([[-np.cos(theta_1), np.cos(theta_1)-np.sin(theta_1)-1, 1+np.sin(theta_1)], 
                           [-np.cos(theta_1), np.cos(theta_1)-np.sin(theta_1)-1, 1+np.sin(theta_1)], 
                           [-np.cos(theta_1), np.cos(theta_1)-np.sin(theta_1)-1, 1+np.sin(theta_1)]])
            k5 = k5/np.linalg.norm(k5)
            k6 = k5.T   #k1,k2,k3,k4的某种组合
            w = k6  #np.cos(theta_1)*k5+np.sin(theta_1)*
            w = 0.95*v2+0.05*w #(np.cos(theta_1)*k2+np.sin(theta_1)*k4)
            w = w/np.linalg.norm(w)
            v.append(w/4)
        
        v = np.array(v)
        
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = tf.zeros(shape=kernel_shape).numpy()
        for i in range(input_shape[-1]):
            temp[:,:,i,:] = v.reshape(self.kernel_size[0], self.kernel_size[1], self.output_dim)
        self.kernel = tf.constant(temp)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='VALID')
        return output

class SphereFeatures(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(SphereFeatures, self).__init__(**kwargs)

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
        c = np.load('vectors_from_PCA.npy')
        c1 = c[0,:].reshape(3, 3)
        c2 = c[1,:].reshape(3, 3)
        c3 = c[2,:].reshape(3, 3)
        thetas = np.arange(0, 2*np.pi, np.pi/4)
        v = []
        for theta_1 in thetas:
            for theta_2 in thetas:
                """
                #甜甜圈中间缩成一点
                w = abs(np.cos(theta_2/2))*(np.cos(theta_2/2)*(np.cos(theta_1)*k2+np.sin(theta_1)*v2)
                                            +np.sin(theta_2/2)*(np.cos(theta_1)*k2+np.sin(theta_1)*v2+k4)/np.sqrt(2)) # 
                """
                #甜甜圈中间略有空隙
                w = 25/49*k2+24/49*(np.cos(theta_2)*(np.cos(theta_1/2+np.pi/16)*k2+np.sin(theta_1/2+np.pi/16)*v22)  #25/49, 24/49
                                            +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                """  
                k5 = np.array([[-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                               [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                               [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)]])
                k5 = k5/np.linalg.norm(k5)
                k6 = k5.T   #k1,k2,k3,k4的某种组合
                k9 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
                k9 = k9/np.sqrt(4)
                w = np.cos(theta_1)*k5+np.sin(theta_1)*k6 #
                """
                #Klein 瓶粘接竖纹
                """
                w = 1/2*(np.cos(theta_1/2-np.pi/2)*k2+np.sin(theta_1/2-np.pi/2)*v2)+1/2*np.sin(theta_1/2)*(np.cos(theta_2)*(np.cos(theta_1/2-np.pi/2)*k2+np.sin(theta_1/2-np.pi/2)*v2)  #25/49, 24/49
                                            +np.sin(theta_2)*k4)
                """
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

class KleinFeatures(layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(KleinFeatures, self).__init__(**kwargs)

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
        c = np.load('vectors_from_PCA.npy')
        c1 = c[0,:].reshape(3, 3)
        c2 = c[1,:].reshape(3, 3)
        c3 = c[2,:].reshape(3, 3)
        thetas = np.arange(0, 2*np.pi, np.pi/4)
        v = []
        for theta_1 in thetas:
            for theta_2 in thetas:
                """
                #甜甜圈中间缩成一点
                w = abs(np.cos(theta_2/2))*(np.cos(theta_2/2)*(np.cos(theta_1)*k2+np.sin(theta_1)*v2)
                                            +np.sin(theta_2/2)*(np.cos(theta_1)*k2+np.sin(theta_1)*v2+k4)/np.sqrt(2)) # 
                """
                #甜甜圈中间略有空隙
                w = 25/49*(np.cos(theta_1/2)*k2+np.sin(theta_1/2)*v21)+24/49*np.cos(theta_1/2)**2*(np.cos(theta_2)*(np.cos(theta_1/2)*k2+np.sin(theta_1/2)*v21/4)  #25/49, 24/49
                                            +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                """  
                k5 = np.array([[-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                               [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)], 
                               [-np.cos(theta_2), np.cos(theta_2)-np.sin(theta_2)-1, 1+np.sin(theta_2)]])
                k5 = k5/np.linalg.norm(k5)
                k6 = k5.T   #k1,k2,k3,k4的某种组合
                k9 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
                k9 = k9/np.sqrt(4)
                w = np.cos(theta_1)*k5+np.sin(theta_1)*k6 #
                """
                #Klein 瓶粘接竖纹
                """
                w = 1/2*(np.cos(theta_1/2-np.pi/2)*k2+np.sin(theta_1/2-np.pi/2)*v2)+1/2*np.sin(theta_1/2)*(np.cos(theta_2)*(np.cos(theta_1/2-np.pi/2)*k2+np.sin(theta_1/2-np.pi/2)*v2)  #25/49, 24/49
                                            +np.sin(theta_2)*k4)
                """
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
        c = np.load('vectors_from_PCA.npy')
        c1 = c[0,:].reshape(3, 3)
        c2 = c[1,:].reshape(3, 3)
        c3 = c[2,:].reshape(3, 3)
        #半个环面，v2对应位置缩成一点
        
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
        """
        thetas = np.arange(0, 9*np.pi/8, np.pi/8)
        v = []
        for theta_1 in thetas:
            theta_2 = 0
            while theta_2/2/np.pi<1-0.001:
                if int(abs(theta_1/np.pi*8-4))==0:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/7
                if int(abs(theta_1/np.pi*8-4))==1:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi*2/11
                        
                if int(abs(theta_1/np.pi*8-4))==2:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi/4
                        
                if int(abs(theta_1/np.pi*8-4))==3:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi*2/5
                        
                if int(abs(theta_1/np.pi*8-4))==4:
                        w = 25/49*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)+24/49*np.sin(theta_1)**2*(np.cos(theta_2)*(np.cos(theta_1)*v22+np.sin(theta_1)*k2)  #25/49, 24/49
                                                    +np.sin(theta_2)*k4) #25/49 k2 + 24/49 …… sphere  #
                        v.append(w/3)
                        theta_2 += np.pi*2
                
        v = np.array(v)
        """
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp = tf.zeros(shape=kernel_shape).numpy()
        for i in range(input_shape[-1]):
            temp[:,:,i,:] = v.reshape(self.kernel_size[0], self.kernel_size[1], self.output_dim)
        self.kernel = tf.constant(temp)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='VALID')
        return output

#for _ in range(100):  #权向量提取
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    #layers.Resizing(64, 64),
    # Normalize.
    norm_layer,
    #ConstantFeatures(64),
    #KleinFeatures(64), 
    #layers.Activation('relu'), 
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  #由于512运算量太大，这里换成128
    layers.Dropout(0.5),
    layers.Dense(num_labels), 
    #layers.Dense(1, activation='sigmoid'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    #loss='binary_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    #callbacks=[SaveWeightsCallback()],
)

y_pred = model.predict(test_spectrogram_ds)

y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

"""
#保存权向量
weights_list_1 = np.array(weights_list_1)
weights_list_2 = np.array(weights_list_2)
weights_list_3 = np.array(weights_list_3)

np.save('weights_list_speech_box_phone_1.npy',weights_list_1)
np.save('weights_list_speech_box_phone_2.npy',weights_list_2)
np.save('weights_list_speech_box_phone_3.npy',weights_list_3)
"""
