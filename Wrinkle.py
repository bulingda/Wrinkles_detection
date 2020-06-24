from PIL import Image
import cv2
import time
import numpy as np
from skimage.filters import frangi, gabor
from skimage import measure, morphology
from Partition import practice14 as pa
#
# class UnionFindSet(object):
#
#     def __init__(self, m):
#         # m, n = len(grid), len(grid[0])
#         self.roots = [i for i in range(m)]
#         self.rank = [0 for i in range(m)]
#         self.count = m
#
#         for i in range(m):
#             self.roots[i] = i
#
#     def find(self, member):
#         tmp = []
#         while member != self.roots[member]:
#             tmp.append(member)
#             member = self.roots[member]
#         for root in tmp:
#             self.roots[root] = member
#         return member
#
#     def union(self, p, q):
#         parentP = self.find(p)
#         parentQ = self.find(q)
#         if parentP != parentQ:
#             if self.rank[parentP] > self.rank[parentQ]:
#                 self.roots[parentQ] = parentP
#             elif self.rank[parentP] < self.rank[parentQ]:
#                 self.roots[parentP] = parentQ
#             else:
#                 self.roots[parentQ] = parentP
#                 self.rank[parentP] -= 1
#             self.count -= 1
#
#
# class Solution(object):
#     def countComponents(self, n, edges):
#         """
#         :type n: int
#         :type edges: List[List[int]]
#         :rtype: int
#         """
#         ufs = UnionFindSet(n)
#         # print ufs.roots
#         for edge in edges:
#             start, end = edge[0], edge[1]
#             ufs.union(start, end)
#
#         return ufs.count
#
#
# # def connected_component(thresh_A):
# #     thresh_A_copy = thresh_A.copy()  # 复制thresh_A到thresh_A_copy
# #     thresh_B = np.zeros(thresh_A.shape).astype(int)  # thresh_B大小与A相同，像素值为0
# #
# #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 3×3结构元
# #
# #     count = []  # 为了记录连通分量中的像素个数
# #     xy_count = []
# #     # 循环，直到thresh_A_copy中的像素值全部为0
# #     while thresh_A_copy.any():
# #         Xa_copy, Ya_copy = np.where(thresh_A_copy > 0)  # thresh_A_copy中值为255的像素的坐标
# #         thresh_B[Xa_copy[0]][Ya_copy[0]] = 1  # 选取第一个点，并将thresh_B中对应像素值改为255
# #
# #         # 连通分量算法，先对thresh_B进行膨胀，再和thresh_A执行and操作（取交集）
# #         for i in range(thresh_A.shape[0]):
# #             dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
# #             print(type(thresh_A), type(dilation_B.astype(int)))
# #             thresh_B = cv2.bitwise_and(np.uint8(thresh_A), dilation_B.astype(int))
# #
# #         # 取thresh_B值为255的像素坐标，并将thresh_A_copy中对应坐标像素值变为0
# #         Xb, Yb = np.where(thresh_B > 0)
# #         thresh_A_copy[Xb, Yb] = 0
# #         xy_count.append(np.c_[Xb, Yb])
# #         # 显示连通分量及其包含像素数量
# #         count.append(len(Xb))
# #         if len(count) == 0:
# #             print("无连通分量")
# #         if len(count) == 1:
# #             print("第1个连通分量为{}".format(count[0]))
# #             print(xy_count[0], "2")
# #             ecc = measure.regionprops(xy_count[0])
# #             print(ecc[0].eccentricity)
# #         if len(count) >= 2:
# #             if (count[-1] - count[-2]) < 2:
# #                 continue
# #             print("第{}个连通分量为{}".format(len(count), count[-1] - count[-2]))
# #             print(xy_count[len(count) - 1][len(xy_count[-2]):len(xy_count[-1])], "2")
# #             ecc = measure.regionprops(xy_count[len(count) - 1][len(xy_count[-2]):len(xy_count[-1])])
# #             print(ecc[0].eccentricity)
# def connected_component(path):
#     img_A = cv2.imread(path)
#     gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
#     ret, thresh_A = cv2.threshold(gray_A, 20, 255, cv2.THRESH_BINARY_INV)  # 灰度图转换成二值图像
#     print(thresh_A, "thresh_A")
#     thresh_A_copy = thresh_A.copy()  # 复制thresh_A到thresh_A_copy
#     thresh_B = np.zeros(thresh_A.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
#
#     count = []  # 为了记录连通分量中的像素个数
#     xy_count = []
#     # 循环，直到thresh_A_copy中的像素值全部为0
#     while thresh_A_copy.any():
#         Xa_copy, Ya_copy = np.where(thresh_A_copy > 0)  # thresh_A_copy中值为255的像素的坐标
#         thresh_B[Xa_copy[0]][Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255
#
#         # 连通分量算法，先对thresh_B进行膨胀，再和thresh_A执行and操作（取交集）
#         for i in range(200):
#             dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
#             thresh_B = cv2.bitwise_and(thresh_A, dilation_B)
#
#         # 取thresh_B值为255的像素坐标，并将thresh_A_copy中对应坐标像素值变为0
#         Xb, Yb = np.where(thresh_B > 0)
#         thresh_A_copy[Xb, Yb] = 0
#         xy_count.append(np.c_[Xb, Yb])
#         # 显示连通分量及其包含像素数量
#         count.append(len(Xb))
#         if len(count) == 0:
#             print("无连通分量")
#         if len(count) == 1:
#             print("第1个连通分量为{}".format(count[0]))
#             # print(np.c_[Xb, Yb], "1")
#             print(xy_count[0], "2")
#
#             ecc = measure.regionprops(xy_count[0])
#             print(ecc[0].eccentricity)
#         if len(count) >= 2:
#             if (count[-1] - count[-2]) < 2:
#                 continue
#             print("第{}个连通分量为{}".format(len(count), count[-1] - count[-2]))
#             # print(np.c_[Xb, Yb], "1")
#             print(xy_count[len(count) - 1][len(xy_count[-2]):len(xy_count[-1])], "2")
#             ecc = measure.regionprops(xy_count[len(count) - 1][len(xy_count[-2]):len(xy_count[-1])])
#             print(ecc[0].eccentricity)


def master_control(image):
    # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation=cv2.INTER_CUBIC)  # 图片分辨率很大是要变小
    b, g, r = cv2.split(image)  # image

    sk_frangi_img = frangi(g, scale_range=(0, 1), scale_step=0.01, beta1=1.5, beta2=0.01)  # 线宽范围，步长，连接程度（越大连接越多），减少程度(越大减得越多)0.015
    sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency=0.35, theta=45)  # 越小越明显
    sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency=0.35, theta=90)
    sk_gabor_img_4, sk_gabor_4 = gabor(g, frequency=0.35, theta=360)  # 横向皱纹
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    all_img = cv2.add(0.1 * sk_gabor_img_2, 0.9 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3
    all_img = morphology.closing(all_img, morphology.disk(1))
    _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
    img1 = all_img
    # print(all_img, all_img.shape, type(all_img))
    # contours, image_cont = cv2.findContours(all_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # all_img = all_img + image_cont
    bool_img = all_img.astype(bool)
    label_image = measure.label(bool_img)
    count = 0

    for region in measure.regionprops(label_image):
        if region.area < 10: #   or region.area > 700
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0

    skel, distance = morphology.medial_axis(all_img.astype(int), return_distance=True)
    skels = morphology.closing(skel, morphology.disk(1))
    trans1 = skels  # 细化
    return skels, count  # np.uint16(skels.astype(int))


def face_wrinkle(path):
    # result = pa.curve(path, backage)
    result = cv2.imread(path)
    img, count = master_control(result)
    print(img.astype(float))
    result[img > 0.1] = 255
    cv2.imshow("result", img.astype(float))
    cv2.waitKey(0)


if __name__ == '__main__':
    path = r"D:\E\document\datas\line\2885.png"
    backage = r'..\Sebum\shape_predictor_81_face_landmarks.dat'

    face_wrinkle(path)

