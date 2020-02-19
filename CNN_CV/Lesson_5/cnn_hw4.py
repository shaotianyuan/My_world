from sklearn.datasets import make_blobs
import numpy as np
import random
import matplotlib.pyplot as plt


def cal_dis(point: list, data: np.array) -> np.array:
    """
    计算点与数据集点欧式距离（支持多维度）
    """
    a = np.sum(np.power(data - point, 2), axis=1)
    return np.sqrt(a)


def nearest_dis(c_centers: np.array, data: np.array):
    """
    质心到数据集点最小距离
    return：最小距离，所属分类
    """
    p_dis = np.zeros((data.shape[0], c_centers.shape[0]))
    for i, j in enumerate(c_centers):
        dis = cal_dis(j, data)
        p_dis[:, i] = dis

    min_dis = np.min(p_dis, axis=1)
    min_idx = np.argmin(p_dis, axis=1)

    return min_dis, min_idx


def creat_centers(data: np.array, k: int):
    """
    knn++的方式创建质心
    1，随机一个数据点作为第一个质心
    2，数据集中的点到不同质心的距离，取最小值，组成dis
    3，dis中较大的点作为新点质心
    4，循环取到k个质心结束
    """
    c_centers = random.choice(data).reshape(1, data.shape[1])

    for i in range(1, k):
        min_dis, min_idx = nearest_dis(c_centers, data)
        l = np.max(min_dis) * 0.70
        # h = np.max(min_dis) * 0.85
        idx = random.choice(np.argwhere(min_dis > l))
        c_centers = np.vstack((c_centers, data[idx]))

        # 轮盘法找到较大值

        # total = np.sum(min_dis)
        # total *= random.random()
        #
        # for i, j in enumerate(min_dis):
        #     total -= j
        #     if total > 0:
        #         continue
        #     c_centers = np.vstack((c_centers, data[i]))
        #     break

    return c_centers


# 随机取质心，做比较用
# def creat_centers_normal(data, k):
#     idx = np.random.choice(data.shape[0], k)
#     c_centers = data[idx]
#     return c_centers

def my_knn_pp(data: np.array, k: int):
    # 生成质心
    c_centers = creat_centers(data, k)
    # 初始化分类
    pre_idx = np.zeros(data.shape[0]) - 1
    n = float('inf')

    plt.ion()
    iter_i = 1

    while n > 2:
        # 最短距离，所属分类
        min_dis, min_idx = nearest_dis(c_centers, data)
        # 计算变动次数
        n = len(np.argwhere(pre_idx != min_idx))

        # 二维数组画散点图
        if data.shape[1] == 2:
            plt.scatter(data[:, 0], data[:, 1], c=pre_idx)
            plt.scatter(c_centers[:, 0], c_centers[:, 1], marker='x', c='red', )
            plt.title(f'K-means ++ iter: No.{iter_i}')
            iter_i += 1
            plt.pause(1)
            if n > 2:
                plt.clf()

        # 更新质心位置
        for i in range(k):
            idx = np.where(min_idx == i)
            c_centers[i] = np.mean(data[idx], axis=0)
        # 更新所属分类
        pre_idx = min_idx

    plt.ioff()
    plt.show()
    return pre_idx


if __name__ == '__main__':
    X, y = make_blobs(1000, 2, 3)
    predict_label = my_knn_pp(X, 3)

