import numpy as np
from NMF_for_olivetti_faces import run_nmf_for_faces
from NMF_item_rec import run_nmf_for_item
from NMF_user_rec import run_nmf_for_user
from NMF_rec import run_nmf_for_rec
# author : Tianyi Yu 
# date : 2022.3.5

# 展示NMF应用于人脸图像处理
run_nmf_for_faces(49, 5, 4) # 参数分别为：选取的特征数、可视化行数、可视化列数

# 电影名称列表（可自行修改）
item = [
    'movie1', 'movie2', 'movie3', 'movie4', 'movie5',
    'movie6', 'movie7', 'movie8', 'movie9', 'movie10',
]
# 用户名称列表（可自行修改）
user = ['user1', 'user2', 'user3', 'user4', 'user5',
        'user6', 'user7', 'user8', 'user9', 'user10',
        'user11', 'user12', 'user13', 'user14', 'user15']
    
# 用户对电影的评分矩阵（可自行修改）
RATE_MATRIX = np.array(
    [[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
     [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
     [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
     [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
     [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
     [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 0, 4, 5, 2, 5],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)

# 展示NMF应用于电影推荐算法
run_nmf_for_item(item, user, RATE_MATRIX)
run_nmf_for_user(item, user, RATE_MATRIX)
run_nmf_for_rec(item, user, RATE_MATRIX)
