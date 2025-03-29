# -*- coding: utf-8 -*-
"""
此程序展示的是从weight到pca图像，以及rip复形计算持续同调的过程。注意这里我们对距离进行了小调整，添加了Sin拟距离进行考量。
此外。本程序还对权向量进行了数据过滤提取。
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import NearestNeighbors
import gudhi as gd
from gudhi.cover_complex import MapperComplex, GraphInducedComplex, NerveComplex
from gudhi import bottleneck_distance
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
import matplotlib
from matplotlib import checkdep_usetex
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams


file_path = 'weights_list_speech_box_phone_1.npy'#replace weight_vectors2, weights_list_speech_box_phone_3
try:
    w = np.load(file_path)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

print(w.shape)


indices = np.arange(w.shape[0]) % 4 == 3
w_20 = w[indices]
#indices = np.arange(w.shape[-1]) % 4 == 3
#w_20 = w_20[...,indices]
w_20 = w_20[:,:,:,0,:]
w_20 = np.expand_dims(w_20, axis=3)

w = np.reshape(w_20, (w_20.shape[0], w_20.shape[1], 
                               w_20.shape[2], w_20.shape[3]*w_20.shape[4]))
w=np.transpose(w, (0, 3, 1,2))


w = np.reshape(w, (w.shape[0]* w.shape[1], w.shape[2], w.shape[3]))
w = np.reshape(w, (w.shape[0], w.shape[1]*w.shape[2]))
print(w.shape)


# Standarize the data

b=np.mean(w, axis=1)
c=np.transpose(b)
d=np.expand_dims(c, axis=1)
w=w-np.tile(d,(1,w.shape[1]))

#print(a)
w =w / np.linalg.norm(w, axis=1, keepdims=True)

# Density Filtration
r=[]
k = 300
p = 0.1

# for row in w:
#     distances = np.linalg.norm(w - row, axis=1)
#     neighbor_id = distances.argsort()[k]
#     #print(distances[neighbor_id])
#     r.append(distances[neighbor_id])

from scipy.spatial.distance import pdist, squareform

# 定义一个函数来计算两个向量之间的距离
def Sin(u, v):
    return np.sqrt(1 - np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))**2)
def Euclid(u, v):
    return np.sqrt(np.sum((u-v)**2))

# 使用pdist和squareform函数计算所有点之间的距离
distances = squareform(pdist(w, metric=Sin))

# 使用NearestNeighbors找到每个点的k个最近邻
nbrs = NearestNeighbors(n_neighbors=k+1, metric='precomputed').fit(distances)
distances, indices = nbrs.kneighbors(distances)  

# Get the distances of the k-th neighbor (index 0 corresponds to the point itself)
r = distances[:, k]
print(len(r))   # size:6400
min_id = np.argsort(r)[:int(len(r)*p)]
#print(min_id)
w_f= w[min_id]
print(w_f.shape)

np.save('vectors_sin_1920_15_3.npy',w_f) #与使用何种度量，向量个数，k值以及选取层数有关


# 使用 PCA
pca = PCA(n_components=9)
vectors_pca = pca.fit_transform(w_f)

# 获取3个坐标轴对应的向量，即PCA的主成分
components = pca.components_

print(components)

# 获取每个主成分解释的方差的百分比
explained_variance_ratio = pca.explained_variance_ratio_


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vectors_pca[:, 0], vectors_pca[:, 1], vectors_pca[:, 2])
plt.title('PCA')
plt.show()

# 创建新的图像
plt.figure()

# 对每个数列计算持续同调并作图
# 计算Rip复形的持续同调
result = ripser(w_f, maxdim=1)

# 在子图中作图
plot_diagrams(result['dgms'], show=True)

