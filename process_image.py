import cv2
import os

path ="..\\waifu2x\\images" #Target of image file
files = os.listdir(path) #Find all files in this file
s=[]

for file in files:
    file_path = path+'\\'+file
    if os.path.exists(file_path):#Check wether this file is exist or not
        img_extension = os.path.splitext(file_path)[-1] #Get filename extension
        if img_extension == '.png':
            s.append(file_path) 
    else:
        print("Not exist")
image=cv2.imread(s[0])
res=cv2.resize(image,(500,500),interpolation=cv2.INTER_CUBIC)
cv2.imshow('iker',res)
cv2.waitKey(0)
cv2.destoryAllWindows()
