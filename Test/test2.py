from sklearn.datasets import make_blobs
import numpy as np
import random
import matplotlib.pyplot as plt


def c_dis(point, data):
    a = np.sum(np.power(data - point, 2), axis=1)
    return np.sqrt(a)


def nearest_dis(c_centers, data):
    p_dis = np.zeros((data.shape[0], c_centers.shape[0]))
    for i, j in enumerate(c_centers):
        dis = c_dis(j, data)
        p_dis[:, i] = dis
    min_dis = np.min(p_dis, axis=1)
    min_idx = np.argmin(p_dis, axis=1)

    return min_dis, min_idx


def creat_centers(data, k):
    c_centers = random.choice(data).reshape(1, data.shape[1])
    for i in range(1, k):
        min_dis, min_idx = nearest_dis(c_centers, data)

        l = np.max(min_dis) * 0.70
        # h = np.max(min_dis) * 0.85
        idx = random.choice(np.argwhere(min_dis > l))
        c_centers = np.vstack((c_centers, data[idx]))

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


# def creat_centers_normal(data, k):
#     idx = np.random.choice(data.shape[0], k)
#     c_centers = data[idx]
#     return c_centers

def my_knn_pp(data, k):
    c_centers = creat_centers(data, k)
    old_idx = np.zeros(data.shape[0]) - 1
    n = float('inf')

    iter_i = 3
    plt.ion()

    while n > 2:
        min_dis, min_idx = nearest_dis(c_centers, data)

        for i in range(k):
            idx = np.where(min_idx == i)
            c_centers[i] = np.mean(data[idx], axis=0)

        n = len(np.argwhere(old_idx != min_idx))
        old_idx = min_idx

        if data.shape[1] == 2:
            plt.scatter(data[:, 0], data[:, 1], c=min_idx)
            plt.scatter(c_centers[:, 0], c_centers[:, 1], marker='x', c='red', )
            plt.title(f'K-means ++ iter: No.{iter_i}')
            iter_i += 1
            plt.pause(1)

            if n > 2:
                plt.clf()

    plt.ioff()
    plt.show()

    return min_idx


if __name__ == '__main__':
    X, y = make_blobs(1000, 2, 3)
    predict_label = my_knn_pp(X, 3)
    print(predict_label)
    print(y)
