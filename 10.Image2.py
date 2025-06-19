# -*- coding: utf-8 -*-
"""
这里添加了OF+NOL，效果也不错
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D, Activation, Layer
import numpy as np
import matplotlib.pyplot as plt 
import math 
from scipy.integrate import tplquad,dblquad,quad
from scipy.linalg import expm

with tf.device('/CPU:0'):

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
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
    
    
    class KleinFeatures(Layer):
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
    
    # 设定种子（可选）
    np.random.seed(42)

    # 1. 生成16个随机的反对称矩阵
    num_kernels = 8
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
                  #[1, -2, 1],
                  #[1, -2, 1]])/np.sqrt(3)
    ]
    A_matrices += [-A for A in A_matrices]

    transformed_matrices = []
    for R in rot_matrices:
        for A in A_matrices:
            A_new = R @ A
            transformed_matrices.append(A_new)
            transformed_matrices.append(A_new.T)

    # 4. 转换为 NumPy 数组
    transformed_matrices = np.array(transformed_matrices)

    # 定义您的模型，包括 OrthogonalFeatures 层
    class OrthogonalFeatures(tf.keras.layers.Layer):
        def __init__(self, transformed_matrices, output_dim, kernel_size=(3, 3), strides=(1, 1), **kwargs):
            super(OrthogonalFeatures, self).__init__(**kwargs)
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
    
    
    
    if __name__ == '__main__':
    
        # get model
        
        #############################  NOL+NOL  ###############################################
        img_input = Input(shape=(32, 32, 3))
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
        
        img_input = Input(shape=(32, 32, 3))
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
        img_input = Input(shape=(32, 32, 3))
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
        
        img_input = Input(shape=(32, 32, 3))
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
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        net = CircleFeatures(64)(img_input)
        #net = KleinFeatures(64)(img_input)
        net = Activation('relu')(net)
        net = MaxPooling2D(2,2)(net)
        #net = Conv2D(64)(net)
        net = CircleOneLayer(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(net)
        #net = KleinOneLayer(64)(net)
        #net = Activation('relu')(net)
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
        
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        net = CircleFeatures(64)(img_input)
        #net = KleinFeatures(64)(img_input)
        net = Activation('relu')(net)
        net = MaxPooling2D(2,2)(net)
        #net = Conv2D(64)(net)
        net = CircleOneLayer(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(net)
        #net = KleinOneLayer(64)(net)
        #net = Activation('relu')(net)
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
        img_input = Input(shape=(32, 32, 3))
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
        
        img_input = Input(shape=(32, 32, 3))
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
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        #net = CircleFeatures(64)(img_input)
        net = KleinFeatures(64)(img_input)
        net = Activation('relu')(net)
        net = MaxPooling2D(2,2)(net)
        #net = Conv2D(64)(net)
        #net = CircleOneLayer(64)(net)
        net = KleinOneLayer(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(net)
        #net = Activation('relu')(net)
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
        
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        #net = CircleFeatures(64)(img_input)
        net = KleinFeatures(64)(img_input)
        net = Activation('relu')(net)
        net = MaxPooling2D(2,2)(net)
        #net = Conv2D(64)(net)
        #net = CircleOneLayer(64)(net)
        net = KleinOneLayer(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(net)
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
        
     #################################OF+NOL#############################################   
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        #net = CircleFeatures(64)(img_input)
        net = OrthogonalFeatures(transformed_matrices=transformed_matrices, output_dim=64)(img_input)
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
                
        model11 = models.Model(img_input, output)
        model11.summary()
                
        model11.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    
        history11 = model11.fit(train_images_noise, train_labels, batch_size=100, epochs=5, 
                           validation_data=(test_images, test_labels))
        
        img_input = Input(shape=(32, 32, 3))
        #net = Conv2D(64)(img_input)
        #net = CircleFeatures(64)(img_input)
        net = OrthogonalFeatures(transformed_matrices=transformed_matrices, output_dim=64)(img_input)
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
                
        model12 = models.Model(img_input, output)
        model12.summary()
                
        model12.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    
        history12 = model12.fit(train_images, train_labels, batch_size=100, epochs=5, 
                           validation_data=(test_images_noise, test_labels))
        
        
        plt.subplot(2,2,1)
        plt.plot(history1.history['val_accuracy'], label = 'NOL+NOL', color = 'blue')
        plt.plot(history3.history['val_accuracy'], label = 'CF+NOL', color = 'purple')
        plt.plot(history5.history['val_accuracy'], label = 'CF+COL', color = 'red')
        plt.plot(history7.history['val_accuracy'], label = 'KF+NOL', color = 'green')
        plt.plot(history9.history['val_accuracy'], label = 'KF+KOL', color = 'orange')
        plt.plot(history11.history['val_accuracy'], label = 'OF+NOL', color = 'black')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
    
        plt.subplot(2,2,2)
        plt.plot(history2.history['val_accuracy'], label = 'NOL+NOL', color = 'blue')
        plt.plot(history4.history['val_accuracy'], label = 'CF+NOL', color = 'purple')
        plt.plot(history6.history['val_accuracy'], label = 'CF+COL', color = 'red')
        plt.plot(history8.history['val_accuracy'], label = 'KF+NOL', color = 'green')
        plt.plot(history10.history['val_accuracy'], label = 'KF+KOL', color = 'orange')
        plt.plot(history12.history['val_accuracy'], label = 'OF+NOL', color = 'black')
        plt.legend(loc='lower right')
    
        plt.subplot(2,2,3)
        plt.plot(history1.history['loss'], label = 'NOL+NOL', color = 'blue')
        plt.plot(history3.history['loss'], label = 'CF+NOL', color = 'purple')
        plt.plot(history5.history['loss'], label = 'CF+COL', color = 'red')
        plt.plot(history7.history['loss'], label = 'KF+NOL', color = 'green')
        plt.plot(history9.history['loss'], label = 'KF+KOL', color = 'orange')
        plt.plot(history11.history['loss'], label = 'OF+NOL', color = 'black')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
    
        plt.subplot(2,2,4)
        plt.plot(history2.history['loss'], label = 'NOL+NOL', color = 'blue')
        plt.plot(history4.history['loss'], label = 'CF+NOL', color = 'purple')
        plt.plot(history6.history['loss'], label = 'CF+COL', color = 'red')
        plt.plot(history8.history['loss'], label = 'KF+NOL', color = 'green')
        plt.plot(history10.history['loss'], label = 'KF+KOL', color = 'orange')
        plt.plot(history12.history['loss'], label = 'OF+NOL', color = 'black')
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
        
        test_loss11, test_acc11 = model11.evaluate(test_images,  test_labels, verbose=1)
    
        print(test_acc11)
        
        test_loss12, test_acc12 = model12.evaluate(test_images_noise,  test_labels, verbose=1)
    
        print(test_acc12)
