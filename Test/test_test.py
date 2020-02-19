from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import random


# x, y = datasets.make_blobs(200, 2, 5)
#
# print(x)
# print(y)

# plt.scatter(x[:,0],x[:,1], c=y)
# plt.show()

def cal_dis(point_1, point_2):
    # 多维度欧式距离
    dis = 0.0
    for a, b in zip(point_1, point_2):
        dis += math.pow(a - b, 2)
    return math.sqrt(dis)


def nearest_dis(point, c_centers):
    # 最近距离
    min_dis = math.inf
    min_c_idx = 0
    for i, c in c_centers:
        dis = cal_dis(point, c)
        if dis < min_dis:
            min_dis = dis
            min_c_idx = i
    return min_c_idx, min_dis


def knn_centers(data, k):
    c_centers = []
    c_centers.append(random.choice(data))
    d = [0.0 for _ in range(len(data))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data):
            idx, d[i] = nearest_dis(point, c_centers)
            total += d[i]

        total *= random.random()
        for i, dis in enumerate(d):
            total -= dis
            if total > 0:
                continue
            c_centers.append(data[i])
            break
    return c_centers
