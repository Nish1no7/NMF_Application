# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib

def run_nmf_for_rec(item, user, RATE_MATRIX):
    nmf_model = NMF(n_components=2)  # 设有2个主题
    item_dis = nmf_model.fit_transform(RATE_MATRIX)
    user_dis = nmf_model.components_
    print(nmf_model.reconstruction_err_)

    rec_mat = np.dot(item_dis, user_dis)
    filter_matrix = RATE_MATRIX < 1e-8
    print('重建矩阵，并过滤掉已经评分的电影:')
    rec_filter_mat = (filter_matrix * rec_mat).T
    print(rec_filter_mat)

    rec_user = 'user11'  # 需要进行推荐的用户
    rec_userid = user.index(rec_user)  # 推荐用户ID
    rec_list = rec_filter_mat[rec_userid, :]  # 推荐用户的电影列表

    print('推荐用户user11的电影:')
    print(np.nonzero(rec_list))


    a = NMF(n_components=2)  # 设有2个主题
    W = a.fit_transform(RATE_MATRIX)
    H = a.components_
    print(a.reconstruction_err_)

    b = NMF(n_components=3)  # 设有3个主题
    W = b.fit_transform(RATE_MATRIX)
    H = b.components_
    print(b.reconstruction_err_)

    c = NMF(n_components=4)  # 设有4个主题
    W = c.fit_transform(RATE_MATRIX)
    H = c.components_
    print(c.reconstruction_err_)

    d = NMF(n_components=5)  # 设有5个主题
    W = d.fit_transform(RATE_MATRIX)
    H = d.components_
    print(d.reconstruction_err_)