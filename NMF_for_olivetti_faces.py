# -*- coding: utf-8 -*-
'''
Author: YuTianyi 18030100394
Date: 2021.11
Using NMF for olivetti face images.  
'''
import numpy as np
import sklearn.decomposition as dp
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState 

def find_martrix_min_value(data_matrix):
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    return min(new_data)

def find_martrix_max_value(data_matrix):
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    return max(new_data)



def plot_gallery(title,images,n_row,n_col):
    plt.figure(figsize=(2.*n_col,2.26*n_row)) #创建图片，并指定图片大小
    plt.suptitle(title,size=18) #设置标题及字号大小
    
    for i in range(0, n_row*n_col):
        comp = images[i,:]
        plt.subplot(n_row,n_col,i+1) #选择绘制的子图
        vmax=max(comp.max(),-comp.min())
        
        plt.imshow(comp.reshape(64, 64),cmap=plt.cm.gray,
                   interpolation='nearest',vmin=-vmax,vmax=vmax) #对数值归一化，并以灰度图形式显示
        plt.xticks(())
        plt.yticks(()) #去除子图的坐标轴标签
    plt.subplots_adjust(0.01,0.05,0.99,0.94,0.04,0.) #对子图位置及间隔调整



def run_nmf_for_faces(n_components, row, col):
    #g = int(math.sqrt(n_components))
    image_shape=(64,64)
    datasets=fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
    #dataset=fetch_olivetti_faces(data_home=None,shuffle=False,random_state=0,download_if_missing=True)
    faces=datasets.data #加载打开数据
    plot_gallery('First centered Olivetti faces',faces[:n_components],row,col)
    estimators=[
            ('Eigenfaces-PCA using randomized SVD',
            dp.PCA(n_components=n_components,whiten=True)),
            ('Non-negative components - NMF',
            dp.NMF(n_components=n_components,init='nndsvda',
                                tol=5e-3))] #NMF和PCA实例化

    for name,estimator in estimators: #分别调用PCA和NMF
        estimator.fit(faces) #调用PCA或NMF提取特征
        components_=estimator.components_ #获取提取的特征
        plot_gallery(name,components_[:n_components],row ,col) #按照固定格式进行排列
    plt.show()

    nmf_model = dp.NMF(n_components=n_components, init='nndsvda', tol = 5e-3)
    H = nmf_model.fit_transform(faces)

    W = nmf_model.components_

    V = np.dot(H, W)

    # 展示原图像
    fig2 = plt.figure(2)
    for i in range(0, row*col):
        ori_img = faces[[i], :].reshape(64, 64)
        #vmax = max(find_martrix_max_value(w_img), -find_martrix_min_value(w_img))
        plt.subplot = fig2.add_subplot(row, col, i+1)
        plt.imshow(ori_img,cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

    # 展示还原后的图像
    fig3 = plt.figure(3)
    for i in range(0, row*col):
        v_img = V[[i], :].reshape(64, 64)
        #vmax = max(find_martrix_max_value(w_img), -find_martrix_min_value(w_img))
        plt.subplot = fig3.add_subplot(row, col, i+1)
        plt.imshow(v_img,cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
    plt.show()

#run_nmf_for_faces(49,4,4)