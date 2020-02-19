# Ransac

import numpy as np
import matplotlib.pyplot as plt
import random

size = 50
error = 100

x = np.linspace(0, 10, size)
y = 3 * x + 10
random_x = [x[i] + random.uniform(-0.5, 0.5) for i in range(size)]
random_y = [y[i] + random.uniform(-0.5, 0.5) for i in range(size)]

# plt.scatter(random_x,random_y)
# plt.show()

for i in range(error):
    random_x.append(random.uniform(0, 20))
    random_y.append(random.uniform(10, 40))

random_x = np.array(random_x)
random_y = np.array(random_y)

# plt.scatter(random_x,random_y)
# plt.show()

from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=True)
reg.fit(random_x.reshape(-1, 1), random_y.reshape(-1, 1))
# slope = reg.coef_
# intercept = reg.intercept_
predict_y = reg.predict(random_x.reshape(-1, 1))

# plt.scatter(random_x,random_y)
# plt.plot(random_x,predict_y,c='red')
# plt.show()

# RANSAC

iterations = 100
tolerent_sigma = 1
thresh_size = 0.5

pretotal = 0
best_slope = -1
best_intercept = 0

plt.ion()
plt.figure()

for i in range(iterations):
    sample_index = random.sample(range(size + error), 2)
    x_1 = random_x[sample_index[0]]
    x_2 = random_x[sample_index[1]]
    y_1 = random_y[sample_index[0]]
    y_2 = random_y[sample_index[1]]

    # y = ax + b
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1

    # calculate inliers
    total_inliers = 0
    for index in range(size + error):
        predict_y = slope * random_x[index] + intercept
        if abs(predict_y - random_y[index]) < tolerent_sigma:
            total_inliers += 1

    if total_inliers > pretotal:
        pretotal = total_inliers
        best_slope = slope
        best_intercept = intercept

    if total_inliers > (size + error) * thresh_size:
        break

    plt.title(f'RANSAC in Linear Regression: Iter{i + 1}, Inliers{pretotal}')
    plt.scatter(random_x, random_y)
    y = best_slope * random_x + best_intercept
    plt.plot(random_x, y, 'black')
    plt.pause(0.2)
    plt.clf()
