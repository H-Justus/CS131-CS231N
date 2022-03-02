# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 22:55
# @Author  : Justus
# @FileName: CS131.py
# @Software: PyCharm

from __future__ import print_function
import random
import numpy as np
from linalg import *
from imageManip import *
import matplotlib.pyplot as plt
import cv2

# ##############PART 1 linear algebra###############
# 输出（4,3）的M，行向量a和列向量b
M = np.linspace(1, 12, 12).reshape((4, 3))
a = np.array([[1, 1, 0]])
b = np.array([[-1], [2], [5]])
print("M = \n", M)
print("The size of M is: ", M.shape)
print()
print("a = \n", a)
print("The size of a is: ", a.shape)
print()
print("b = \n", b)
print("The size of b is: ", b.shape)

# 输出a点乘b
aDotB = dot_product(a, b)
print()
print(aDotB)
print("The size is: ", aDotB.shape)

# 计算(a * b) * (M * a.T)
print()
ans = complicated_matrix_function(M, a, b)
print(ans)
print("The size is: ", ans.shape)

print()
M_2 = np.array(range(4)).reshape((2, 2))
a_2 = np.array([[1, 1]])
b_2 = np.array([[10, 10]]).T
print(M_2.shape)
print(a_2.shape)
print(b_2.shape)
print()
ans = complicated_matrix_function(M_2, a_2, b_2)
print(ans)
print("The size is: ", ans.shape)

# 输出前k个奇异值
print()
only_first_singular_value = get_singular_values(M, 1)
print(only_first_singular_value)
first_two_singular_values = get_singular_values(M, 2)
print(first_two_singular_values)
assert only_first_singular_value[0] == first_two_singular_values[0]

# 进行特征值分解，返回最大的k个特征值和相应的特征向量
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
val, vec = get_eigen_values_and_vectors(M[:, :3], 1)
print("First eigenvalue =", val[0])
print()
print("First eigenvector =", vec[0])
print()
assert len(vec) == 1
val, vec = get_eigen_values_and_vectors(M[:, :3], 2)
print("Eigenvalues =", val)
print()
print("Eigenvectors =", vec)
assert len(vec) == 2

# ##############PART 2 image Manipulation###############
image1_path = './image1.jpg'
image2_path = './image2.jpg'


def display(img, title=None, camp=None):
    # Show image
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(img, cmap=camp)
    plt.show()


# 加载并显示头像
image1 = load(image1_path)
image2 = load(image2_path)
display(image1)
display(image2)

# 图像调暗
new_image = dim_image(image1)
display(new_image)

# 转换为灰度图
grey_image = convert_to_grey_scale(image1)
display(grey_image, "gray", "gray")

# 剔除一个RGB通道
without_red = rgb_exclusion(image1, 'R')
without_blue = rgb_exclusion(image1, 'B')
without_green = rgb_exclusion(image1, 'G')
display(without_red, "Below is the image without the red channel.")
display(without_green, "Below is the image without the green channel.")
display(without_blue, "Below is the image without the blue channel.")

# 返回一个LAB通道
image_l = lab_decomposition(image1, 'L')
image_a = lab_decomposition(image1, 'A')
image_b = lab_decomposition(image1, 'B')
display(image_l, "Below is the image with only the L channel.")
display(image_a, "Below is the image with only the A channel.")
display(image_b, "Below is the image with only the B channel.")

# 返回一个HSV 通道
image_h = hsv_decomposition(image1, 'H')
image_s = hsv_decomposition(image1, 'S')
image_v = hsv_decomposition(image1, 'V')
display(image_h, "Below is the image with only the H channel.")
display(image_s, "Below is the image with only the S channel.")
display(image_v, "Below is the image with only the V channel.")
# skimage和cv2处理结果有不同
img = cv2.imread(image1_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
cv2.waitKey()

# 创建一个新图像，使图像的左半边是image1的左半边，而图像的右半边是image2的右半边。排除给定图像的指定通道。
image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)
assert int(np.sum(image_mixed)) == 76421

# 左上象限:移除'R'通道。右上角象限:调暗。左下角象限:点亮:右下象限:移除'R'通道。
image_mix_qdr = mix_quadrants(image1)
display(image_mix_qdr)
