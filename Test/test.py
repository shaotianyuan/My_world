import numpy as np
import matplotlib.pyplot as plt
import cv2

# a = cv2.imread('/Users/sty/PycharmProjects/My_world/CNN_CV/Lesson_2/Data/noise_lenna.jpg')
#
# b = a.shape
# print(b)
# #
# for s in [a, b, c]:
#     k = np.zeros((s.shape[0] - 2, s.shape[1] - 2), dtype='uint8')
#     for i in range(s.shape[0]-2):
#         for j in range(s.shape[1]-2):
#             b = s[i:i+3,j:j+3].ravel()
#             k[i][j] = np.median(b)
#     # print(k)
#     BGR_new.append(k)
#     # print(BGR_new)

data = np.arange(16).reshape(4, 4)
a = np.median(data)
print(data)
print(a)


