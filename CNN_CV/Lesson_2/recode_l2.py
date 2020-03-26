import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/sty/PycharmProjects/My_world/CNN_CV/Lesson_1/Data/lenna.jpg', 1)

def my_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def my_show2(img1,img2):
    plt.subplot(121)
    my_show(img1)
    plt.subplot(122)
    my_show(img2)
    plt.show()

# 一阶导图像
# 高斯滤波
g_img = cv2.GaussianBlur(img, (11, 11), 2)
# my_show(g_img)

# 高斯算子
kernel_1d = cv2.getGaussianKernel(11, 2)
print(kernel_1d)
# 拿到算子，卷积sepFilter2D（一阶高斯参数：分别对X轴方向和Y轴方向求导）
g1_img = cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)
# my_show(g1_img)

# laplacian
# 二阶求导算子
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
lap_img1 = cv2.filter2D(img, -1, kernel)
# my_show(lap_img1)

kernel_strong = np.array([[1,1,1],[1,-8,1],[1,1,1]])
lap_img2 = cv2.filter2D(img, -1, kernel_strong)

# plt.figure(figsize=(10,5),dpi=120)
plt.subplot(121)
my_show(lap_img1)
plt.subplot(122)
my_show(lap_img2)
plt.show()

# 图片锐化，相当于在原图上加一层边缘(kernel中间像素+1)
kernel_r = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])
r_img = cv2.filter2D(img, -1, kernel_r)
plt.subplot(121)
my_show(img)
plt.subplot(122)
my_show(r_img)
plt.show()

kernel_r2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
r_img2 = cv2.filter2D(img, -1, kernel_r2)
plt.subplot(121)
my_show(img)
plt.subplot(122)
my_show(r_img2)
plt.show()

# sobel
# 一阶求导算子

# y轴求导
kernel_sx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sx_img = cv2.filter2D(img, -1, kernel_sx)
# x轴求导
kernel_sy = np.array([[-1,0,-1],[-2,0,2],[1,0,1]])
sy_img = cv2.filter2D(img, -1, kernel_sy)
plt.subplot(121)
my_show(sx_img)
plt.subplot(122)
my_show(sy_img)
plt.show()

# medianblur

n_img = cv2.imread('/Users/sty/PycharmProjects/My_world/CNN_CV/Lesson_2/Data/noise_lenna.jpg')
md_img = cv2.medianBlur(n_img, 7)
gd_img = cv2.GaussianBlur(n_img,(3,3),2)

plt.subplot(131)
my_show(n_img)
plt.subplot(132)
my_show(md_img)
plt.subplot(133)
my_show(gd_img)
plt.show()

# Harris Corner

# def my_show_gray(img):
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cmap='gray')

# my_show_gray(img)
# plt.show()

img_harris = cv2.cornerHarris(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 2, 3, 0.03)
threshold = np.max(img_harris) * 0.02

img[img_harris > threshold] = [0, 0, 255]
my_show(img)


img_t = cv2.imread('/Users/sty/PycharmProjects/My_world/CNN_CV/Lesson_2/Data/test_corner.jpg')
my_show(img_t)

img_tgray = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
# img_tgray = cv2.dilate(img_tgray, None)  # 扩展一下
img_tH = cv2.cornerHarris(img_tgray, 2, 3, 0.03)
# img_tH = cv2.dilate(img_tH, None) # 扩展一下

threshold2 = np.max(img_tH) * 0.02
img_t[img_tH > threshold2] = [0, 0, 255]
my_show(img_t)

# SIFT

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img)

print(len(kp))
kp, des = sift.compute(img, kp)
print(des.shape)

img_sift = cv2.drawKeypoints(img, kp, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10,10),dpi=100)
my_show(img_sift)
