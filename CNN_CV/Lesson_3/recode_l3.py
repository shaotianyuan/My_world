import numpy as np
import random
import matplotlib.pyplot as plt

# 线性函数模型：预测Y
def inference(w, b, x):
    pred_y = w * x + b
    return pred_y

# loss funciton
def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0
    for i in range(len(x_list)):
        avg_loss += 0.5 * (w * x_list[i] + b - gt_y_list[i]) ** 2
    avg_loss /= len(gt_y_list)
    return avg_loss

# 单一样本带来的梯度
def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db

# 全部样本（batchsize）为w，b带来的更新
def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pred_y = inference(w, b, batch_x_list[i])
        dw, db = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size

    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

# 训练样本
def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w, b = 0, 0
    w_list = []
    b_list = []
    z_list = []
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]

        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        z = eval_loss(w,b,x_list,gt_y_list)
        print('w:{0},b:{1}'.format(w, b))
        print('loss is {}'.format(eval_loss(w,b,x_list,gt_y_list)))
        w_list.append(w)
        b_list.append(b)
        z_list.append(z)

    return w_list,b_list,z_list

# 制造数据
def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(100, 200) + random.random()

    num_sample = 100
    x_list = []
    y_list = []
    print(w, b)
    for i in range(num_sample):
        x = random.randint(0, 100) + random.random()
        y = w * x + b + random.random() * random.randint(-10, 10)

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list

x_list, y_list = gen_sample_data()

plt.figure()
plt.scatter(x_list, y_list)
plt.show()

# train(x_list, y_list, 60, 0.0001, 100)
w, b, z = train(x_list, y_list, 60, 0.00001, 100)
plt.scatter(w, z)
plt.show()
plt.scatter(b, z)
plt.show()