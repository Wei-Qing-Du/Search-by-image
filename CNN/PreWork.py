import os
import numpy as np
import math
import tensorflow as tf
import re
import matplotlib.pyplot as plt
import cv2

# coding=utf-8
"""
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
"""


train_dir = r'C:\Users\Z97MX-GAMING\Desktop\train'

frist_type = []
label_frist_type = []

second_type = []
label_second_type = []

third_type = []
label_third_type = []

fourth_type = []
label_fourth_type = []

fifth_type = []
label_fifth_type = []

sixth_type = []
label_sixth_type = []

seventh_type = []
label_seventh_type = []

eightth_type = []
label_eightth_type = []

ninth_type = []
label_ninth_type = []

tenth_type = []
label_tenth_type = []





#step1:Get path fo images from the Image_to_tfrecords.py
    #Store all path of images to the list and give labels that save to the label of list 
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir):
        pattern = re.compile("^\d")
        data_type = pattern.findall(file)

        if (data_type[0] == '0'):
            frist_type.append(file_dir+'\\'+file)
            label_frist_type.append(0)

        elif (data_type[0] == '1'):
            second_type.append(file_dir+'\\'+file)
            label_second_type.append(1)

        elif (data_type[0] == '2'):
            third_type.append(file_dir+'\\'+file)
            label_third_type.append(2)

        elif (data_type[0] == '3'):
            fourth_type.append(file_dir+'\\'+file)
            label_fourth_type.append(3)

        elif (data_type[0] == '4'):
            fifth_type.append(file_dir+'\\'+file)
            label_fifth_type.append(4)

        elif (data_type[0] == '5'):
            sixth_type.append(file_dir+'\\'+file)
            label_sixth_type.append(5)

        elif (data_type[0] == '6'):
            seventh_type.append(file_dir+'\\'+file)
            label_seventh_type.append(6)

        elif (data_type[0] == '7'):
            eightth_type.append(file_dir+'\\'+file)
            label_eightth_type.append(7)

        elif (data_type[0] == '8'):
            ninth_type.append(file_dir+'\\'+file)
            label_ninth_type.append(8)

        elif (data_type[0] == '9'):
            tenth_type.append(file_dir+'\\'+file)
            label_tenth_type.append(9)            
        print("Write file %s" %file)

#step2: Combine paht and labels into one list 
    image_list = np.hstack((frist_type,second_type,third_type,fourth_type,fifth_type,sixth_type,
                            seventh_type,eightth_type,ninth_type,tenth_type)) # Let difference list to the same array with horizon.
    label_list = np.hstack((label_frist_type,label_second_type,label_third_type,label_fourth_type,label_fifth_type,label_sixth_type,
                            label_seventh_type,label_eightth_type,label_ninth_type,label_tenth_type))

    #利用shuffle,转置，随机打乱
    temp = np.array([image_list,label_list])    #Trun into 2x array
    temp = temp.transpose()     #转置
    np.random.shuffle(temp)     #The array or list to be shuffled
    #print(temp)
    #从打乱的temp中再取出list（img和lab）
    #image_list = list(temp[:,0])
    #label_list = list(temp[:,1])
    #label_list = [int(i) for i in label_list]
    #return  image_list,label_list

    all_image_list = list(temp[:,0])   
    all_label_list = list(temp[:,1])   

    #Divide list to two parts whose one is for training and other is for testing
    #ratio is ration of testing
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  #Get test data
    n_train = n_sample - n_val    #Get num of train data

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
#    tra_labels = [int(float(i)) for i in tra_labels]    #Transform to int
    tra_labels = [int(i) for i in tra_labels]

    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
#    val_labels = [int(float(i)) for i in val_labels]    #Transform to int
    val_labels = [int(i) for i in val_labels]
    #print(val_labels)

    return tra_images,tra_labels,val_images,val_labels
#--------------------------生成batch------------------------

#step1：Make the list introduce to the get_batch(), and change type，make a queue for img and lab.
#So use tf.train.slice_input_producer(),and use tf.read_file() read images from queue.
#   capacity：size of queue

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #Use tf.cast() to change type
    image = tf.cast(image,tf.string)    
    label = tf.cast(label,tf.int32)    

#    print(label)
    # tf.train.slice_input_producer to make tensor
    #make an input queue
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()to read images

    #step2:Decode the images，that must use same type。
#    image = tf.image.decode_image(image_contents,channels=3)
    image =tf.image.decode_jpeg(image_contents, channels=3)#cv2.imread(image_contents) 

    #step3:data preprocession
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)   #Standard image after be resized
#    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.

    #step4:Make the batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch ,label_batch = tf.train.batch([image,label],
                                              batch_size = batch_size,
                                              num_threads= 32,
                                              capacity =capacity)
    #Reshape label，row size is [batch_size]
    label_batch =tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    #image_batch = tf.cast(image_batch,tf.float32)     #显示灰度图像

    return image_batch,label_batch
    # Get two batches that are introduced into the CNN.



def PreWork():
    # See performance of prwork
    IMG_W = 32
    IMG_H = 32
    BATCH_SIZE = 6
    CAPACITY = 64
    #train_dir = 'F:/Python/PycharmProjects/DeepLearning/CK+_part'
    # image_list, label_list, val_images, val_labels = get_file(train_dir)
    image_list, label_list = get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(label_batch.shape)
    lists = ('0', '1', '2', '3', '4', '5','6','7','8','9')
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # Creat the thread manager
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                # Get two batches to be showed on the image.
                img, label = sess.run([image_batch, label_batch])  # Get img and label from the session.
                # img = tf.cast(img, tf.uint8)

                for j in np.arange(BATCH_SIZE):
                    print('label: %d' % label[j])
                    plt.imshow(img.eval())
                    title = lists[int(label[j])]
                    plt.title(title)
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    #PreWork()
