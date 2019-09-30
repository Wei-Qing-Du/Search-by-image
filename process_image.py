import cv2
import os
path ="..\\waifu2x\\images" #Target of image file
files = os.listdir(path) #Find all files in this file
s=[]
for file in files:
    if os.path.exists(file):#Check wether this file is exist or not
        file = open(file, 'r')#Read the file
        print(file.readline())
        file.close()
    else:
        print("Not exist")
#image=cv2.imread('test.jpg')
#res=cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
#cv2.imshow('iker',res)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destoryAllWindows()
