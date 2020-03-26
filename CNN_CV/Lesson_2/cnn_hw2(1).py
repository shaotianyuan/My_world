import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_medianblur(img, n):
    BGR_new = []
    row = img.shape[0] - n + 1
    col = img.shape[1] - n + 1
    for s in cv2.split(img):
        k = np.zeros((row, col), dtype=s.dtype)
        for i in range(row):
            for j in range(col):
                b = s[i:i+n,j:j+n]
                k[i][j] = np.median(b)
        BGR_new.append(k)

    return cv2.merge((BGR_new[0],BGR_new[1],BGR_new[2]))

noise_img = cv2.imread('/Users/sty/PycharmProjects/My_world/CNN_CV/Lesson_2/Data/noise_lenna.jpg')

mymedianblur_img = my_medianblur(noise_img, 3)
medianblur = cv2.medianBlur(noise_img, 3)

plt.subplot(131)
plt.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.imshow(cv2.cvtColor(mymedianblur_img, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.imshow(cv2.cvtColor(medianblur, cv2.COLOR_BGR2RGB))
plt.show()