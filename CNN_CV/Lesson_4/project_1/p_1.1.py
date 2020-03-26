import cv2
import numpy as np
import sys


class My_stitch():
    def __init__(self, ratio=0.85, windows=200):
        self.ratio = ratio  # knnmatch两点的差异比率
        self.windows = windows  # 合并后窗口大小

    def registration(self, img1, img2):
        # sift特征点与描述子
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # 暴力匹配，knnmatch找到欧式距离最接近的k个点
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        # 第一个点要比第二个点 领先的够多，否则 两个人都不够格，好的关键点具有独立性
        good_points = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append(m1)
        # 匹配图
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_points, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        # 利用Ransac找到变换矩阵
        image1_kp = np.float32([kp1[i.queryIdx].pt for i in good_points])  # keypoint.pt 坐标
        image2_kp = np.float32([kp2[i.trainIdx].pt for i in good_points])
        H, mask = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        # 参数1：需要变换的图的关键点坐标
        # 参数2：对比的图的关键点坐标
        # 参数3：方法
        # 参数4：方法的阈值
        return H

    def creat_mask(self, img1, img2, version):
        # panorama size
        height_panorama = img1.shape[0]
        width_panorama = img1.shape[0] + img2.shape[1]

        # window size
        offset = self.windows // 2
        barrier = img1.shape[1] - offset

        mask = np.zeros((height_panorama, width_panorama))
        if version == 0:
            mask[:, barrier - offset: barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset), (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset: barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset), (height_panorama, 1))
            mask[:, barrier + offset:] = 1

        return cv2.merge((mask, mask, mask))

    def blending(self, img1, img2):
        # 变换矩阵
        H = self.registration(img1, img2)
        # 左蒙板，右蒙板
        mask1 = self.creat_mask(img1, img2, 0)
        mask2 = self.creat_mask(img1, img2, 1)
        # panorama size
        height_panorama = img1.shape[0]
        width_panorama = img1.shape[0] + img2.shape[1]

        # 拼接
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:img1.shape[0], 0: img1.shape[1], :] = img1
        panorama1 *= mask1
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        result = panorama1 + panorama2

        # 去黑边
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

def main(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final = My_stitch().blending(img1, img2)
    cv2.imwrite('panorama.jpg', final)

if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except IndexError:
        print("Please input two source images: ")
        print("For example: python p_1.1.py '/project_1/test1.jpeg' '/project_1/test2.jpeg'")