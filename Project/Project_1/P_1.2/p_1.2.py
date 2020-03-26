import cv2
import matplotlib.pyplot as plt

def registration_knn(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matcher_point = matcher.knnMatch(des2, des1, k=2)

    good_points = []
    for m1, m2 in matcher_point:
        if m1.distance < 0.85 * m2.distance:
            good_points.append(m1)
    good_points = sorted(good_points, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img2, kp2, img1, kp1, good_points[:60], None, flags=2)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

def main():
    img1 = cv2.imread('test1.jpeg')
    img2 = cv2.imread('test4.png')
    registration_knn(img1, img2)

main()