# -*- coding: utf-8 -*-
"""
这里是MNIST数据集的复现，使用的tensorflow自带数据集。对于CIFAR10数据集，只需再代码正文第一行中更改即可，并同时更新输入形状
对于各层的定义，这里是采取的将lOVE等人的结果先计算，再放进去。实际上，直接积分即可。
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D, Activation, Layer
import numpy as np
import matplotlib.pyplot as plt 
import math 
from scipy.integrate import tplquad,dblquad,quad


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()  #cifar10

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Gauss Noise
train_images_noise = train_images
test_images_noise = test_images
for i in range(len(train_images[:,0,0])):
    train_images_noise[i,:,:] = train_images[i,:,:]+np.random.normal(np.random.normal(.2,.04), np.random.chisquare(.04), 
                                                  train_images[0,:,:].shape)
for i in range(len(test_images[:,0,0])):    
    test_images_noise[i,:,:] = test_images[i,:,:]+np.random.normal(np.random.normal(.2,.04), np.random.chisquare(.04), 
                                                 test_images[0,:,:].shape)

class_names = ['0','1','2','3','4','5','6','7','8','9']

class CircleFeatures(Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(CircleFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        chi_shape = (1, 1, input_shape[-1], self.output_dim)
        chi_init = np.arange(0,2*math.pi,2/self.output_dim*math.pi)
        temp_1 = tf.zeros(shape=chi_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            temp_1[:,:,i,:] = chi_init
        chi = tf.constant(temp_1, name='chi')
        chi = tf.Variable(chi, trainable=False)
        v1 = tf.constant([-1, 0, 1], shape=(3, 1), dtype=tf.float32, name='v1')
        v2 = tf.constant([1, 1, 1], shape=(3, 1), dtype=tf.float32, name='v2')
        cos_chi = tf.math.cos(chi)
        sin_chi = tf.math.sin(chi)
        c1 = tf.constant(8/27, shape=(), dtype=tf.float32)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp_2 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            for j in range(self.output_dim): 
                temp_2[:,:,i,j] = tf.add(tf.matmul(tf.matmul(v1, cos_chi[:,:,i,j]), tf.reshape(v2, (1, 3))), 
                tf.matmul(tf.matmul(v2, sin_chi[:,:,i,j]), tf.reshape(v1, (1, 3)))).numpy()
        t_chi = tf.constant(temp_2)
        self.kernel = tf.scalar_mul(c1, t_chi)
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='SAME')
        return output

    
class KleinFeatures(Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(KleinFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        chi_shape = (1, 1, input_shape[-1], int(math.sqrt(self.output_dim)))  # theta_shape = chi_shape
        chi_init = np.arange(0,math.pi,1/int(math.sqrt(self.output_dim))*math.pi)
        theta_init = np.arange(0,2*math.pi,2/int(math.sqrt(self.output_dim))*math.pi)
        temp_1 = tf.zeros(shape=chi_shape, dtype=tf.float32).numpy()
        temp_2 =temp_1
        for i in range(input_shape[-1]):
            temp_1[:,:,i,:] = chi_init
            temp_2[:,:,i,:] = theta_init
        chi = tf.constant(temp_1, name='chi')
        chi = tf.Variable(chi, trainable=False)
        theta = tf.constant(temp_2, name='theta')
        theta = tf.Variable(theta, trainable=False)
        v1 = tf.constant([-1, 0, 1], shape=(3, 1), dtype=tf.float32, name='v1')
        v2 = tf.constant([1, 1, 1], shape=(3, 1), dtype=tf.float32, name='v2')
        cos_chi = tf.math.cos(chi)
        sin_chi = tf.math.sin(chi)
        cos_theta = tf.math.cos(theta)
        sin_theta = tf.math.sin(theta)
        c0 = tf.constant(-100/243, shape=(), dtype=tf.float32)
        c1 = tf.constant(8/27, shape=(), dtype=tf.float32)
        c2 = tf.constant(32/81, shape=(), dtype=tf.float32)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        sqrt_kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], int(math.sqrt(self.output_dim)))
        temp_3 = tf.zeros(shape=sqrt_kernel_shape, dtype=tf.float32).numpy()
        temp_4 = temp_3
        temp_5 = temp_4
        for i in range(input_shape[-1]):
            for j in range(int(math.sqrt(self.output_dim))): 
                temp_3[:,:,i,j] = tf.add(tf.matmul(tf.matmul(v1, cos_chi[:,:,i,j]), tf.reshape(v2, (1, 3))), 
                tf.matmul(tf.matmul(v2, sin_chi[:,:,i,j]), tf.reshape(v1, (1, 3)))).numpy()
                temp_4[:,:,i,j] = tf.matmul(tf.matmul(v2, cos_theta[:,:,i,j]), tf.reshape(v2, (1, 3))).numpy()
                temp_5[:,:,i,j] = tf.matmul(tf.matmul(v2, sin_theta[:,:,i,j]), tf.reshape(v2, (1, 3))).numpy()
        t_chi = tf.constant(temp_3)
        t_chi_2 = tf.math.multiply(t_chi, t_chi)
        cos_theta_L = tf.constant(temp_4)
        sin_theta_L = tf.constant(temp_5)
        temp_6 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        temp_7 = temp_6
        temp_8 = temp_7
        temp_9 = tf.ones(shape=sqrt_kernel_shape, dtype=tf.float32)
        for k in range(self.output_dim):
            temp_6[:,:,:,k] = tf.scalar_mul(c0, 
                                            tf.math.multiply(cos_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             temp_9[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
            temp_7[:,:,:,k] = tf.scalar_mul(c1, 
                                            tf.math.multiply(sin_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             t_chi[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
            temp_8[:,:,:,k] = tf.scalar_mul(c2, 
                                            tf.math.multiply(cos_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             t_chi_2[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
        t0 = tf.constant(temp_6)
        t1 = tf.constant(temp_7)
        t2 = tf.constant(temp_8)
        self.kernel = tf.add(t0, tf.add(t1, t2))
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='SAME')
        return output
    
class CircleOneLayer(Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), distance=1/4, **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.distance = distance
        super(CircleOneLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        chi_shape = (1, 1, input_shape[-1], self.output_dim)
        chi_init = np.arange(0,2*math.pi,2/self.output_dim*math.pi)
        temp_1 = tf.zeros(shape=chi_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            temp_1[:,:,i,:] = chi_init
        chi = tf.constant(temp_1, name='chi')
        chi = tf.Variable(chi, trainable=False)
        v1 = tf.constant([-1, 0, 1], shape=(3, 1), dtype=tf.float32, name='v1')
        v2 = tf.constant([1, 1, 1], shape=(3, 1), dtype=tf.float32, name='v2')
        cos_chi = tf.math.cos(chi)
        sin_chi = tf.math.sin(chi)
        c1 = tf.constant(8/27, shape=(), dtype=tf.float32)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        temp_2 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            for j in range(self.output_dim): 
                temp_2[:,:,i,j] = tf.add(tf.matmul(tf.matmul(v1, cos_chi[:,:,i,j]), tf.reshape(v2, (1, 3))), 
                tf.matmul(tf.matmul(v2, sin_chi[:,:,i,j]), tf.reshape(v1, (1, 3)))).numpy()
        t_chi = tf.constant(temp_2)
        #self.kernel = tf.scalar_mul(c1, t_chi)
        temp_3 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        shape_temp = (self.kernel_size[0], self.kernel_size[1])
        for i in range(input_shape[-1]):
            for j in range(self.output_dim): 
                if abs(i/input_shape[-1]-j/self.output_dim) < self.distance or abs(i/input_shape[-1]-j/self.output_dim) > 1-self.distance:
                    temp_3[:,:,i,j] = tf.ones(shape=shape_temp, dtype=tf.float32).numpy()
        c2 = tf.constant(temp_3)
        self.kernel = tf.multiply(c2, tf.scalar_mul(c1, t_chi))
    
    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='SAME')
        return output
    
class KleinOneLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), distance=2/3, **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.distance = distance
        super(KleinOneLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        chi_shape = (1, 1, input_shape[-1], int(math.sqrt(self.output_dim)))  # theta_shape = chi_shape
        chi_init = np.arange(0,math.pi,1/int(math.sqrt(self.output_dim))*math.pi)
        theta_init = np.arange(0,2*math.pi,2/int(math.sqrt(self.output_dim))*math.pi)
        temp_1 = tf.zeros(shape=chi_shape, dtype=tf.float32).numpy()
        temp_2 =temp_1
        for i in range(input_shape[-1]):
            temp_1[:,:,i,:] = chi_init
            temp_2[:,:,i,:] = theta_init
        chi = tf.constant(temp_1, name='chi')
        chi = tf.Variable(chi, trainable=False)
        theta = tf.constant(temp_2, name='theta')
        theta = tf.Variable(theta, trainable=False)
        v1 = tf.constant([-1, 0, 1], shape=(3, 1), dtype=tf.float32, name='v1')
        v2 = tf.constant([1, 1, 1], shape=(3, 1), dtype=tf.float32, name='v2')
        cos_chi = tf.math.cos(chi)
        sin_chi = tf.math.sin(chi)
        cos_theta = tf.math.cos(theta)
        sin_theta = tf.math.sin(theta)
        c0 = tf.constant(-100/243, shape=(), dtype=tf.float32)
        c1 = tf.constant(8/27, shape=(), dtype=tf.float32)
        c2 = tf.constant(32/81, shape=(), dtype=tf.float32)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim)
        sqrt_kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], int(math.sqrt(self.output_dim)))
        temp_3 = tf.zeros(shape=sqrt_kernel_shape, dtype=tf.float32).numpy()
        temp_4 = temp_3
        temp_5 = temp_4
        for i in range(input_shape[-1]):
            for j in range(int(math.sqrt(self.output_dim))): 
                temp_3[:,:,i,j] = tf.add(tf.matmul(tf.matmul(v1, cos_chi[:,:,i,j]), tf.reshape(v2, (1, 3))), 
                tf.matmul(tf.matmul(v2, sin_chi[:,:,i,j]), tf.reshape(v1, (1, 3)))).numpy()
                temp_4[:,:,i,j] = tf.matmul(tf.matmul(v2, cos_theta[:,:,i,j]), tf.reshape(v2, (1, 3))).numpy()
                temp_5[:,:,i,j] = tf.matmul(tf.matmul(v2, sin_theta[:,:,i,j]), tf.reshape(v2, (1, 3))).numpy()
        t_chi = tf.constant(temp_3)
        t_chi_2 = tf.math.multiply(t_chi, t_chi)
        cos_theta_L = tf.constant(temp_4)
        sin_theta_L = tf.constant(temp_5)
        temp_6 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        temp_7 = temp_6
        temp_8 = temp_7
        temp_9 = tf.ones(shape=sqrt_kernel_shape, dtype=tf.float32)
        for k in range(self.output_dim):
            temp_6[:,:,:,k] = tf.scalar_mul(c0, 
                                            tf.math.multiply(cos_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             temp_9[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
            temp_7[:,:,:,k] = tf.scalar_mul(c1, 
                                            tf.math.multiply(sin_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             t_chi[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
            temp_8[:,:,:,k] = tf.scalar_mul(c2, 
                                            tf.math.multiply(cos_theta_L[:,:,:,k//int(math.sqrt(self.output_dim))], 
                                                             t_chi_2[:,:,:,k%int(math.sqrt(self.output_dim))])).numpy()
        t0 = tf.constant(temp_6)
        t1 = tf.constant(temp_7)
        t2 = tf.constant(temp_8)
        #self.kernel = tf.add(t0, tf.add(t1, t2))
        
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
        
        k1 = tf.zeros(shape=kernel_shape, dtype=tf.float32).numpy()
        for i in range(input_shape[-1]):
            for j in range(self.output_dim):
                if D(i%8 * math.pi/int(math.sqrt(input_shape[-1])), i//int(math.sqrt(input_shape[-1])) *2* math.pi/int(math.sqrt(input_shape[-1])), j%int(math.sqrt(self.output_dim)) * math.pi/int(math.sqrt(self.output_dim)), j//int(math.sqrt(self.output_dim))*2*math.pi/int(math.sqrt(self.output_dim)))<=self.distance:  
                    k1[:,:,i,j] = tf.ones(shape=(self.kernel_size[0], self.kernel_size[1]), dtype=tf.float32).numpy()
        t3 = tf.constant(k1)
        self.kernel = tf.multiply(tf.add(t0, tf.add(t1, t2)), t3)
        
        def call(self, inputs):
            output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='SAME')
            return output


if __name__ == '__main__':

    # get model
    
    #############################  NOL+NOL  ###############################################
    img_input = Input(shape=(28, 28, 1))   #（32，32，3） for cifar10
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(img_input)
    #net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model1 = models.Model(img_input, output)
    model1.summary()
            
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history1 = model1.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images, test_labels))
    
    img_input = Input(shape=(28, 28, 1))
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(img_input)
    #net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model2 = models.Model(img_input, output)
    model2.summary()
            
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history2 = model2.fit(train_images, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images_noise, test_labels))
    
    #############################  CF+NOL  ###############################################
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model3 = models.Model(img_input, output)
    model3.summary()
            
    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history3 = model3.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images, test_labels))
    
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model4 = models.Model(img_input, output)
    model4.summary()
            
    model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history4 = model4.fit(train_images, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images_noise, test_labels))
    
    #############################  CF+COL  ###############################################
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    #net = Conv2D(64)(net)
    net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model5 = models.Model(img_input, output)
    model5.summary()
            
    model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history5 = model5.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images, test_labels))
    
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    net = CircleFeatures(64)(img_input)
    #net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    #net = Conv2D(64)(net)
    net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model6 = models.Model(img_input, output)
    model6.summary()
            
    model6.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history6 = model6.fit(train_images, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images_noise, test_labels))
    
    #############################  KF+NOL  ###############################################
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    #net = CircleFeatures(64)(img_input)
    net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model7 = models.Model(img_input, output)
    model7.summary()
            
    model7.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history7 = model7.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images, test_labels))
    
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    #net = CircleFeatures(64)(img_input)
    net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Conv2D(64, (3,3), activation='relu', padding = 'same')(net)
    #net = CircleOneLayer(64)(net)
    #net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model8 = models.Model(img_input, output)
    model8.summary()
            
    model8.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history8 = model8.fit(train_images, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images_noise, test_labels))
    
    #############################  KF+KOL  ###############################################
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    #net = CircleFeatures(64)(img_input)
    net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    #net = Conv2D(64)(net)
    #net = CircleOneLayer(64)(net)
    net = KleinOneLayer(64)(net)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model9 = models.Model(img_input, output)
    model9.summary()
            
    model9.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history9 = model9.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images, test_labels))
    
    img_input = Input(shape=(28, 28, 1))
    #net = Conv2D(64)(img_input)
    #net = CircleFeatures(64)(img_input)
    net = KleinFeatures(64)(img_input)
    net = Activation('relu')(net)
    net = MaxPooling2D(2,2)(net)
    #net = Conv2D(64)(net)
    #net = CircleOneLayer(64)(net)
    net = KleinOneLayer(64)(net)
    net = MaxPooling2D(2,2)(net)
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(10)(net)
    output = net
            
    model10 = models.Model(img_input, output)
    model10.summary()
            
    model10.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history10 = model10.fit(train_images, train_labels, batch_size=100, epochs=5, 
                       validation_data=(test_images_noise, test_labels))
    
    plt.subplot(2,2,1)
    plt.plot(history1.history['val_accuracy'], label = 'NOL+NOL', color = 'blue')
    plt.plot(history3.history['val_accuracy'], label = 'CF+NOL', color = 'purple')
    plt.plot(history5.history['val_accuracy'], label = 'CF+COL', color = 'red')
    plt.plot(history7.history['val_accuracy'], label = 'KF+NOL', color = 'green')
    plt.plot(history9.history['val_accuracy'], label = 'KF+KOL', color = 'orange')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.subplot(2,2,2)
    plt.plot(history2.history['val_accuracy'], label = 'NOL+NOL', color = 'blue')
    plt.plot(history4.history['val_accuracy'], label = 'CF+NOL', color = 'purple')
    plt.plot(history6.history['val_accuracy'], label = 'CF+COL', color = 'red')
    plt.plot(history8.history['val_accuracy'], label = 'KF+NOL', color = 'green')
    plt.plot(history10.history['val_accuracy'], label = 'KF+KOL', color = 'orange')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.subplot(2,2,3)
    plt.plot(history1.history['loss'], label = 'NOL+NOL', color = 'blue')
    plt.plot(history3.history['loss'], label = 'CF+NOL', color = 'purple')
    plt.plot(history5.history['loss'], label = 'CF+COL', color = 'red')
    plt.plot(history7.history['loss'], label = 'KF+NOL', color = 'green')
    plt.plot(history9.history['loss'], label = 'KF+KOL', color = 'orange')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    plt.subplot(2,2,4)
    plt.plot(history2.history['loss'], label = 'NOL+NOL', color = 'blue')
    plt.plot(history4.history['loss'], label = 'CF+NOL', color = 'purple')
    plt.plot(history6.history['loss'], label = 'CF+COL', color = 'red')
    plt.plot(history8.history['loss'], label = 'KF+NOL', color = 'green')
    plt.plot(history10.history['loss'], label = 'KF+KOL', color = 'orange')
    plt.legend(loc='lower right')
    plt.show()
    
    test_loss1, test_acc1 = model1.evaluate(test_images,  test_labels, verbose=1)

    print(test_acc1)
    
    test_loss2, test_acc2 = model2.evaluate(test_images_noise,  test_labels, verbose=1)

    print(test_acc2)
    
    test_loss3, test_acc3 = model3.evaluate(test_images,  test_labels, verbose=1)

    print(test_acc3)
    
    test_loss4, test_acc4 = model4.evaluate(test_images_noise,  test_labels, verbose=1)

    print(test_acc4)
    
    test_loss5, test_acc5 = model5.evaluate(test_images,  test_labels, verbose=1)

    print(test_acc5)
    
    test_loss6, test_acc6 = model6.evaluate(test_images_noise,  test_labels, verbose=1)

    print(test_acc6)
    
    test_loss7, test_acc7 = model7.evaluate(test_images,  test_labels, verbose=1)

    print(test_acc7)
    
    test_loss8, test_acc8 = model8.evaluate(test_images_noise,  test_labels, verbose=1)

    print(test_acc8)
    
    test_loss9, test_acc9 = model9.evaluate(test_images,  test_labels, verbose=1)

    print(test_acc9)
    
    test_loss10, test_acc10 = model10.evaluate(test_images_noise,  test_labels, verbose=1)

    print(test_acc10)