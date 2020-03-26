import numpy as np
import matplotlib.pyplot as plt
import random

# 制造数据
def random_data():
    w = random.randint(0,10) + random.random()
    b = random.randint(0,5) + random.random()
    print(w, b)
    x_array = np.random.randint(1,100,(100,1)) * np.random.random((100,1))
    y_array = w * x_array + b + np.random.random((100,1)) * np.random.randint(-100, 100,(100,1))
    return x_array, y_array

# 预期函数
def inference(w, b, x_array):
    pred_y_array = w * x_array + b
    return pred_y_array

# 损失函数
def eval_loss(w, b, x_array, gt_y_array):
    loss_array = 0.5 * (w * x_array + b - gt_y_array) ** 2
    return np.mean(loss_array)

def step_gradient(batch_x_array, batch_y_array, w, b, lr):
    pred_y_array = inference(w, b, batch_x_array)
    diff = pred_y_array - batch_y_array
    dw = np.mean(diff * batch_x_array)
    db = np.mean(diff)
    w -= dw * lr
    b -= db * lr
    return w, b

def train(x_array, y_array, batch_size, lr, max_iter):
    w, b = 0, 0
    w_list = []
    b_list = []
    z_list = []
    for i in range(max_iter):
        batch_index = np.random.choice(x_array.shape[0],batch_size)
        batch_x = x_array[batch_index]
        batch_y = y_array[batch_index]
        w, b = step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0},b:{1}'.format(w, b))
        print('loss is {}'.format(eval_loss(w, b, x_array, y_array)))
        # time.sleep(0.1)
        z = eval_loss(w, b, x_array, y_array)
        w_list.append(w)
        b_list.append(b)
        z_list.append(z)

    return w_list,b_list,z_list

def run():
    x_array, y_array = random_data()
    w, b, j =train(x_array, y_array, 50, 0.0001, 100)
    y = x_array * w[99] + b[99]

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.plot(x_array, y)
    plt.scatter(x_array, y_array)
    plt.subplot(132)
    plt.scatter(w, j)
    plt.subplot(133)
    plt.scatter(b, j)
    plt.show()

for i in range(5):
    run()


