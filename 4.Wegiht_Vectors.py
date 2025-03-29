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

for _ in range(100):  #权向量提取
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


#保存权向量
weights_list_1 = np.array(weights_list_1)
weights_list_2 = np.array(weights_list_2)
weights_list_3 = np.array(weights_list_3)

np.save('weights_list_speech_box_phone_1.npy',weights_list_1)
np.save('weights_list_speech_box_phone_2.npy',weights_list_2)
np.save('weights_list_speech_box_phone_3.npy',weights_list_3)

