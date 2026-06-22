from PIL import Image
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import glob
import filecmp
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
import win32com.client as win32


def rotate_image(image_path, angle, save_path):
    # 打开图片
    image = Image.open(image_path)

    # 旋转图片
    rotated_image = image.rotate(angle)

    # 保存旋转后的图片
    rotated_image.save(save_path)

def least_squares_fit(x, y):
    # 计算平均值以便于后续计算
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    # 计算总差异和总X平方差
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    # 最小二乘法参数估计
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = y_bar - slope * x_bar

    return slope, intercept

# def delete_edge(arr,axis,x,y,w,h):
#     num = 20
#     up = y - h/2 +num
#     down = y + h/2-num
#     left = x - w/2+num
#     right = x + w/2-num
#     # 上
#     if axis == 1:
#         i = 0
#         while i <len(arr):
#             if arr[i][1] < up:
#                 arr = np.delete(arr, i, axis=0)
#             else:
#                 i = i+1
#         print(arr)
#     # 下
#     elif axis == 2:
#         i = 0
#         while i < len(arr):
#             if arr[i][1] > down:
#                 arr = np.delete(arr, i, axis=0)
#             else:
#                 i = i + 1
#     # 左
#     elif axis == 3:
#         i = 0
#         while i < len(arr):
#             if arr[i][0] < left:
#                 arr = np.delete(arr, i, axis=0)
#             else:
#                 i = i + 1
#     # 右
#     elif axis == 4:
#         i = 0
#         while i < len(arr):
#             if arr[i][0] > right:
#                 arr = np.delete(arr, i, axis=0)
#             else:
#                 i = i + 1
#
    # return arr



def delete_noise(arr,axis):
    i = 0
    while i < len(arr)-1:
        if axis == 'y' :
            if abs(arr[i + 1][1] - arr[i][1]) > 100:
                arr=np.delete(arr, i + 1,axis=0)
            else:
                i += 1
        elif axis == 'x':
            if abs(arr[i + 1][0] - arr[i][0]) > 100:
                arr=np.delete(arr, i + 1,axis=0)
            else:
                i += 1
    return arr


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def x_max(arr):
    coords = np.unravel_index(arr[:, 0].argmax(), arr[:, 0].shape)
    return coords
def x_min(arr):
    coords = np.unravel_index(arr[:, 0].argmin(), arr[:, 0].shape)
    return coords
def y_max(arr):
    coords = np.unravel_index(arr[:, 1].argmax(), arr[:, 1].shape)
    return coords
def y_min(arr):
    coords = np.unravel_index(arr[:, 1].argmin(), arr[:, 1].shape)
    return coords
def Default():
    return "Invalid"
arr_dict = {
    1: x_max,
    2: x_min,
    3: y_max,
    4: y_min
}
def angle_change(angle):
    if angle < 0:
        angle += 180
    return angle

def arr_change(list,type):
    arr = np.array(list)
    return cal_angle(arr, arr_dict.get(type, Default)(arr))

def find_duplicates(arr1, arr2):
    # 将两个数组转换为集合，并找出重复元素

    set1 = set(tuple(element) for element in arr1)
    set2 = set(tuple(element) for element in arr2)

    # 找到共同元素
    duplicates = np.array(list(set1.intersection(set2)))
    # 将重复元素转换回数组
    return np.array(list(duplicates))


def fun(imagefile,num=0,cw = 0):
    num = num % 4
    lst1 = []
    lst2 = []
    lst3 = []
    lst4 = []
    model = YOLO(r'D:\desktop\train\model\best_wan12.2.pt')
    results = model(imagefile, max_det=1)
    for result in results:
        masks = result.masks
        boxes = result.boxes
    box = boxes.xywhn.tolist()
    x = box[0][0] * 640
    y = box[0][1] * 640
    w = box[0][2] * 640
    h = box[0][3] * 640

    num0 = w * 0.12
    up = y - h / 2 + num0
    down = y + h / 2 - num0
    left = x - w / 2 + num0
    right = x + w / 2 - num0

    masks_lst = masks.data.tolist()
    for i in range(len(masks_lst[0])):
        for j in range(len(masks_lst[0][i]) - 1):
            if masks_lst[0][i][j] != masks_lst[0][i][j + 1] and masks_lst[0][i][j] == 0 and left < j < right:
                lst1.append([j + 1, i])
            if masks_lst[0][i][j] != masks_lst[0][i][j + 1] and masks_lst[0][i][j] == 1 and left < j < right:
                lst2.append([j, i])
    for i in range(len(masks_lst[0][0])):
        for j in range(len(masks_lst[0]) - 1):
            if masks_lst[0][j][i] != masks_lst[0][j + 1][i] and masks_lst[0][j][i] == 0 and up < j < down:
                lst3.append([i, j + 1])
            if masks_lst[0][j][i] != masks_lst[0][j + 1][i] and masks_lst[0][j][i] == 1 and up < j < down:
                lst4.append([i, j])

    arr1 = delete_noise(np.array(lst1), 'x')
    arr2 = delete_noise(np.array(lst2), 'x')
    arr3 = delete_noise(np.array(lst3), 'y')
    arr4 = delete_noise(np.array(lst4), 'y')
    #
    my_array1 = arr1[:x_min(arr1)[0]]  # 左上
    my_array2 = arr1[x_min(arr1)[0]:]  # 左下
    my_array3 = arr2[:x_max(arr2)[0]]  # 右上
    my_array4 = arr2[x_max(arr2)[0]:]  # 右下
    my_array5 = arr3[:y_min(arr3)[0]]  # 左上
    my_array6 = arr3[y_min(arr3)[0]:]  # 右上
    my_array7 = arr4[:y_max(arr4)[0]]  # 左下
    my_array8 = arr4[y_max(arr4)[0]:]  # 右下
    #
    # x1 = [arr2[i][0] for i in range(len(arr2))]
    # y1 = [-arr2[i][1] for i in range(len(arr2))]
    # plt.scatter(x1,y1)
    #
    # plt.show()
    # 右下1
    line1 = find_duplicates(my_array4, my_array8)
    # 左下2
    line2 = find_duplicates(my_array2, my_array7)
    # 左上3
    line3 = find_duplicates(my_array1, my_array5)
    # 右上4
    line4 = find_duplicates(my_array3, my_array6)
    angle = []
    # x1 = [line1[i][0] for i in range(len(line1))]
    # y1 = [-line1[i][1] for i in range(len(line1))]
    # x2 = [line2[i][0] for i in range(len(line2))]
    # y2 = [-line2[i][1] for i in range(len(line2))]
    # x3 = [line3[i][0] for i in range(len(line3))]
    # y3 = [-line3[i][1] for i in range(len(line3))]
    # x4 = [line4[i][0] for i in range(len(line4))]
    # y4 = [-line4[i][1] for i in range(len(line4))]
    # plt.scatter(x1,y1)
    # plt.scatter(x2, y2)
    # plt.scatter(x3, y3)
    # plt.scatter(x4, y4)
    # plt.show()
    if len(line1) > 2 and len(line2) > 2 and len(line3) > 2 and len(line4) > 2:
        slope1, intercept1 = least_squares_fit(line1[:, 0], line1[:, 1])
        angle.append(angle_change(-math.atan(slope1) * 180.0 / math.pi))

        slope2, intercept2 = least_squares_fit(line2[:, 0], line2[:, 1])
        angle.append(angle_change(-math.atan(slope2) * 180.0 / math.pi))

        slope3, intercept3 = least_squares_fit(line3[:, 0], line3[:, 1])
        angle.append(angle_change(-math.atan(slope3) * 180.0 / math.pi))

        slope4, intercept4 = least_squares_fit(line4[:, 0], line4[:, 1])
        angle.append(angle_change(-math.atan(slope4) * 180.0 / math.pi))
        # print([angle_change(angle[i]) for i in range(4)])
        if abs(abs(angle[2] - angle[0])) < 10 and abs(abs(angle[3] - angle[1])) < 10:
            if cw == 0:
                return angle[num]
            elif cw == 1:
                return swapPositions(angle,1,3)[num]
        else:
            if num == 0 or num == 2:
                return 0
            else:
                return 90
    else:
        if num == 0 or num == 2:
            return 0
        else:
            return 90


def fun2(imagefile, num=0,cw = 0):
    num = num % 4
    angle = fun(imagefile, num, cw)
    if 20 < angle < 70 or 110 < angle < 160:
        return angle
    else:
        if 0 <= angle <= 20 or 90 <= angle <= 110:
            rotate_angle = 45
        else:
            rotate_angle = -45
        rotate_image(imagefile, rotate_angle, r'D:\desktop\train\temp.png')
        angle2 = fun(r'D:\desktop\train\temp.png', num,cw)
        return angle2 - rotate_angle


def changexls2xlsx(filename):
    if filename != '0':
        excel = win32.gencache.EnsureDispatch('Excel.Application')
        wb = excel.Workbooks.Open(filename)
        wb.SaveAs(filename + "x", FileFormat=51)  # FileFormat = 51 is for .xlsx extension
        wb.Close()  # FileFormat = 56 is for .xls extension
        excel.Application.Quit()
        os.remove(filename)



# print(fun2(r'D:\desktop\train\photo\2024年10月31日14-40-26光谱.jpg',5))
# print(fun2(r'D:\desktop\train\photo\2024年10月31日20-29-56光谱.jpg',0))
# print(fun2(r'D:\desktop\train\photo\2024年10月31日14-40-50光谱.jpg',5))

# print(fun(r'D:\desktop\train\temp.png',0))
# 2024年10月31日14-42-36光谱