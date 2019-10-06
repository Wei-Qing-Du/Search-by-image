import cv2
import os

# coding=utf-8
path = input("Please input the path:")#Target of image file
store_path = "new_image"
files = os.listdir(path) #Find all files in this file
s=[]

for file in files:
    file_path = path+'\\'+file
    if os.path.exists(file_path):#Check wether this file is exist or not
        img_extension = os.path.splitext(file_path)[-1] #Get filename extension
        if img_extension == '.jpg':
            s.append(file_path) 
    else:
        print("Not exist")

if not os.path.isdir(store_path):
    os.makedirs(store_path)

for read in s:
    image=cv2.imread(read, cv2.IMREAD_UNCHANGED )
    res=cv2.resize(image,(64,64),interpolation=cv2.INTER_CUBIC)#Resize the images
    
    read = read.split("\\")[-1] #Get file name
    print(store_path+'\\'+read+"  sotring............")
    cv2.imwrite(store_path+'\\'+read, res)