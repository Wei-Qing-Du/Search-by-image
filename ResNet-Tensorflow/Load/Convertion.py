# -*- coding: utf-8 -*-
from keras.preprocessing.image import save_img
import numpy as np
import pickle
import argparse
import os
import threading as th

NUM_CLASSES = 10

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--p', type=str, help='Enter dataset')
parser.add_argument('--typ', type=str, help='Enter train or test')
arg = parser.parse_args()

datSetpath = arg.p

train = "train1"
test = "test1"
if arg.typ == "train":
    datSetpath += (train + "/")
else:
    datSetpath += (test + "/")

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def buildfolder(path):
    for i in range(NUM_CLASSES):
        if not os.path.isdir(path + str(i)):
            os.mkdir(path + str(i))

def CovetimgfromDataSet(Type = "train",  index = 1):    
    if Type == "train":
        dataName = "/data_batch_" + str(index)  
        Xtr = unpickle(arg.p + dataName)
        print( dataName + " is loading... from Thread id:" + str(th.current_thread())+"\n")
        
        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            
            if  not os.path.isdir(datSetpath):
                os.mkdir(datSetpath)
                
            buildfolder(datSetpath)
            cls = str(Xtr['labels'][i])
            picName = cls+ "/" +str(Xtr['labels'][i]) + '_' + str(i + (index - 1)*10000) + '.jpg'
            imgpath = datSetpath + picName
            save_img(imgpath, img)
            imgpath = ""
            print( dataName + " is loaded... from Thread id:" + str(th.current_thread())+"\n")
        print("train_batch loaded.")
    else:
        print("test_batch is loading...")
        testXtr = unpickle(arg.p + "test_batch")
        for i in range(0, 10000):
            img = np.reshape(testXtr['data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            
            if  not os.path.isdir(datSetpath):
                os.mkdir(datSetpath)
                
            buildfolder(datSetpath)
            cls = str(testXtr['labels'][i])
            picName = cls+ "/" +str(testXtr['labels'][i]) + '_' + str(i + (index - 1)*10000) + '.jpg'
            imgpath = datSetpath + picName
            save_img(imgpath, img)
            imgpath = ""
        print("test_batch loaded.")

def main(typ):
    t_list = []
   
    if typ == "train":
        t1 = th.Thread(target = CovetimgfromDataSet, args=("train", 1))
        t_list.append(t1)
        
        t2 = th.Thread(target = CovetimgfromDataSet, args=("train", 2))
        t_list.append(t2)
        t3 = th.Thread(target = CovetimgfromDataSet, args=("train", 3))
        t_list.append(t3)
        t4 = th.Thread(target = CovetimgfromDataSet, args=("train", 4))
        t_list.append(t4)
        t5 = th.Thread(target = CovetimgfromDataSet, args=("train", 5))
        t_list.append(t5)
        
    
        for t in t_list:
            t.start()
        for t in t_list:
            t.join()
    else:
        CovetimgfromDataSet("test")
    
       
if __name__ == "__main__":
    main(arg.typ)

