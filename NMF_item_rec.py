# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt



def run_nmf_for_item(item, user, RATE_MATRIX):
    nmf_model = NMF(n_components=2) # 设有2个主题
    item_dis = nmf_model.fit_transform(RATE_MATRIX)
    user_dis = nmf_model.components_

    print('用户的主题分布:')
    print(user_dis)
    print('电影的主题分布:')
    print(item_dis)

    plt1 = plt
    plt1.plot(item_dis[:, 0], item_dis[:, 1], 'ro')
    plt1.xlim((-1, 3))
    plt1.ylim((-1, 3))
    plt1.title(u'the distribution of items (NMF)')#设置图的标题

    count = 1
    zipitem = zip(item, item_dis)#把电影标题和电影的坐标联系在一起

    for item in zipitem:
        item_name = item[0]
        data = item[1]
        plt1.text(data[0], data[1], item_name,
                horizontalalignment='center',
                verticalalignment='top')

    plt1.show()

#run_nmf_for_item(item, user, RATE_MATRIX)