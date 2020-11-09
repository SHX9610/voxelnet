import torch
import numpy as np
import cv2

def Slice(x):
    '''
    :param x:  numpy
    :return:  切好等分的tensor(有编号）
    '''
    h,w = x.shape
    item_h = int(h/3)
    item_w = int(w/3)
    image_list = []
    for i in range(3):
        for j in range(3):
            box = (j*item_w,(j+1)*item_w,i*item_h,(i+1)*item_h)
            image_list.append(x[box[2]:box[3],box[0]:box[1]])
            print(box)
    image_array= np.array(image_list)
    return image_array


def Add(image_array):
    '''
    :param image_array:  矩阵 形状  （9，n，n）
    :return:    拼接后得图 （整体的）
    '''

    r = []
    for i in range(0,9,3):
        a = image_array[i]
        b = image_array[i+1]
        c = image_array[i+2]
        r.append(np.concatenate((a,b,c),axis=1))
    image_cat = np.concatenate((r[0],r[1],r[2]),axis=0)

    return image_cat


if __name__ == '__main__':
    x = np.load('E:\peizhunsd\data\\ref\\A02_1.npy')
    image_array = Slice(x)
    print(image_array.shape)
    image_cat = Add(image_array)
    print(image_cat.shape)



