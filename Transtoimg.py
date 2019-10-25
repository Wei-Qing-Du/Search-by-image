import pickle
import tensorflow as tf
import numpy as np
import cv2
import imageio
import os


def unpickle(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding= 'latin1')
        fo.close()
    return data

#Creat file of train and test
if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("test"):
    os.mkdir("train")


#Train
for j in range(1, 6):#train data 1 to 6
    dataName = "data_batch_" + str(j)
    Xtr = unpickle(dataName)
    print(dataName + " is loading")

    for i in range(0, 10000):
        img =  np.reshape(Xtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = 'train/' + str(Xtr['labels'][i])  + '_' + str(i + (j - 1) * 10000) + '.jpg'
        imageio.imwrite(picName, img)
    print(dataName + " loaded.")
testXtr = unpickle("test_batch")

#Test
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imwrite(picName, img)
print("test_batch loaded.")