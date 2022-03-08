# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


def run_nmf_for_user(item, user, RATE_MATRIX):
    nmf_model = NMF(n_components=2) # 设有2个主题
    item_dis = nmf_model.fit_transform(RATE_MATRIX)
    user_dis = nmf_model.components_

    print('用户的主题分布:')
    print(user_dis)
    print('电影的主题分布:')
    print(item_dis)

    user_dis = user_dis.T
    plt1 = plt
    plt1.plot(user_dis[:, 0], user_dis[:, 1], 'bo')
    plt1.xlim((-1, 3))
    plt1.ylim((-1, 3))
    plt1.title(u'the distribution of user (NMF)')#设置图的标题

    zipuser = zip(user, user_dis)#把用户名称标题和用户的坐标联系在一起
    for user in zipuser:
        user_name = user[0]
        data = user[1]
        plt1.text(data[0], data[1], user_name,
                horizontalalignment='center',
                verticalalignment='top')

    plt1.show()

#run_nmf_for_user(item, user, RATE_MATRIX)