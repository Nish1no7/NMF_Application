# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.decomposition import NMF

r = 2    # W矩阵的列数以及H矩阵的行数
k = 100  # 迭代次数上限
E = 1e-5 # 误差阈值

def train(V):
    n, m = np.shape(V)
    # 初始化W和H矩阵
    W = np.random.random((n, r))
    H = np.random.random((r, m))
    for x in range(k):
        V_pre = np.dot(W, H)
        Error = np.linalg.norm(V - V_pre)/(n*m)
        if Error < E:
            break
        
        # 更新W矩阵
        a = np.dot(V, H.T)           # 分子 n*r
        b = np.dot(np.dot(W,H), H.T) # 分母 n*r
        W[b != 0] = (W * a / b)[b != 0]
        
        # 更新H矩阵
        c = np.dot(W.T, V)            # 分子 r*m
        d = np.dot(np.dot(W.T, W), H) # 分母 r*m
        H[d != 0] = (H * c / d)[d != 0]
        
    print("Error = ", Error)
    print("round = ", x)
    return W, H 

if __name__ == "__main__":

    # 生成目标矩阵V
    origin_W = np.random.randint(0,9,size=(100, 2))
    origin_H = np.random.randint(0,9,size=(2, 80))
    V = np.dot(origin_W, origin_H) 
    X = np.array([[1,1,5,2,3], [0,6,2,1,1], [3, 4,0,3,1], [4, 1,5,6,3]])
    time_start = time.time()
    W, H = train(X)
    time_end = time.time()
    print("Init_X = \n",X)
    print("W = \n",W)
    print("H = \n",H)
    print("W*H = \n",np.dot(W,H))
    print("totally time cost: ", time_end - time_start)
