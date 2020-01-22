import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs,make_classification

# 期望函数h(x)
def inference(theta, data):
    pred_y = 1 / (1 + np.exp(np.dot(theta,data)))
    return pred_y

# 损失函数J(theta)
def eval_loss(theta, data, label):
    pred_y = inference(theta, data)
    loss_value = (0 - label) * np.log(pred_y) - (1 - label) * np.log(1 - pred_y)
    return np.mean(loss_value)

# 梯度下降
def step_gradient(pred_y, label, theta, data, lr):
    for i in range(len(theta)):
        diff = np.mean((pred_y - label) * data[i])
        theta[i] -= lr * diff
    return theta

# 训练模型
def train(data, label, batch_size, lr, max_iter):
    data_t = np.c_[np.ones(data.shape[0]), data].T
    theta = np.zeros(data_t.shape[0])
    for i in range(max_iter):
        batch_index = np.random.choice(len(label), batch_size)
        batch_data = data_t[:, batch_index]
        batch_label = label[batch_index]
        batch_pred_y = inference(theta, batch_data)
        theta = step_gradient(batch_pred_y, batch_label, theta, batch_data, lr)
        loss_value = eval_loss(theta, data_t, label)
        print('------第{}次迭代------'.format(i))
        print('theta:{}'.format(theta))
        print('loss_value:{}'.format(loss_value))

    return theta



# 制造数据
def run():
    x, y = make_blobs(1000, 2, 2)
    theta = train(x, y, 160, 0.0001, 100)
    plt.scatter(x[:, 0],x[:, 1],marker='o', c=y)
    d = (theta[0] + theta[1] * x[:, 0]) / -theta[2]
    plt.plot(x[:, 0], d)
    plt.show()

for i in range(5):
    run()

