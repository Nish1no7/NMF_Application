# -*- coding: utf-8 -*-
import numpy as np
import time
r = 10    # W矩阵的列数以及H矩阵的行数
k = 100  # 迭代次数上限
E = 1e-5 # 误差阈值

def train(V):
    n, m = np.shape(V)
    # 初始化W和H矩阵
    W = np.mat(np.random.random((n, r)))
    H = np.mat(np.random.random((r, m)))

    for x in range(k):
        V_pre = W * H
        Error = np.linalg.norm(V - V_pre)/(n*m)
        if Error < E:
            break
        
        # 更新W矩阵
        a = V * H.T     # 分子 n*r
        b = W * H * H.T # 分母 n*r
        for i in range(n):
            for j in range(r):
                if b[i,j] != 0:
                    W[i,j] = W[i,j] * a[i,j] / b[i,j]
        
        # 更新H矩阵
        c = W.T * V     # 分子 r*m
        d = W.T * W * H # 分母 r*m
        for i in range(r):
            for j in range(m):
                if d[i,j] != 0:
                    H[i,j] = H[i,j] * c[i,j] / d[i,j]
        
    print("Error = ", Error)
    return W,H 

if __name__ == "__main__":

    # 生成目标矩阵V
    origin_W = np.random.randint(0,9,size=(100, 2))
    origin_H = np.random.randint(0,9,size=(2, 80))
    V = np.dot(origin_W, origin_H) 
    time_start = time.time()
    W, H = train(V)
    time_end = time.time()
    print("Init_V = ",V)
    print("\n")
    print("W = ",W)
    print("\n")
    print("H = ",H)
    print("\n")
    print("W*H = ",W*H)
    print("totally time cost: ", time_end - time_start)