import os
import numpy as np
import math
import tensorflow as tf

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

#将上图中的两种花的0,1分类改成sunflowers和roses两个文件夹
#pre_dir = r"D:\PyCharm\KinZhang_First_ImageDetection\generate_data"

#os.renames(pre_dir+'/'+"1",pre_dir+'/'+"sunflowers")
#os.renames(pre_dir+'/'+"0",pre_dir+'/'+"roses")
#os.renames(r"D:\PyCharm\KinZhang_First_ImageDetection\generate_data\0",r"D:\PyCharm\KinZhang_First_ImageDetection\generate_data\sunflowers")

train_dir = r'C:\Users\Z97MX-GAMING\Desktop\generate_data'

roses = []
label_roses = []
sunflowers = []
label_sunflowers = []

#step1:Get path fo images from the Image_to_tfrecords.py
    #Store all path of images to the list and give labels that save to the label of list 
def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'/roses'):
        roses.append(file_dir+'/roses'+'/'+file)
        label_roses.append(0)
    for file in os.listdir(file_dir+'/sunflowers'):
        sunflowers.append(file_dir+'/sunflowers'+'/'+file)
        label_sunflowers.append(1)

    print("There are %d roses\nThere are %d sunflowers\n"%(len(roses),len(sunflowers)),end="")

#step2: Combine paht and labels into one list 
    image_list = np.hstack((roses,sunflowers)) # Let difference list to the same array with horizon.
    label_list = np.hstack((label_roses,label_sunflowers))

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
    image = tf.image.decode_jpeg(image_contents, channels=3)

    #step3:data preprocession
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)   #Standard image after be resized
#    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉105行（标准化）和 121行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
#    image = tf.image.per_image_standardization(image)
#具体解释地址看该链接代码：https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/input_data.py

    #step4:Make the batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch ,label_batch = tf.train.batch([image,label],
                                              batch_size = batch_size,
                                              num_threads= 32,
                                              capacity =capacity)
    #Reshape label，row size is [batch_size]
    label_batch =tf.reshape(label_batch,[batch_size])
#    image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image_batch,tf.float32)     #显示灰度图像

    return image_batch,label_batch
    # Get two batches that are introduced into the CNN.



def PreWork():
    # 对预处理的数据进行可视化，查看预处理的效果
    IMG_W = 256
    IMG_H = 256
    BATCH_SIZE = 6
    CAPACITY = 64
    train_dir = 'F:/Python/PycharmProjects/DeepLearning/CK+_part'
    # image_list, label_list, val_images, val_labels = get_file(train_dir)
    image_list, label_list = get_file(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(label_batch.shape)
    lists = ('angry', 'disgusted', 'fearful', 'happy', 'sadness', 'surprised')
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                # 提取出两个batch的图片并可视化。
                img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label
                # img = tf.cast(img, tf.uint8)
                '''
                1、range()返回的是range object，而np.arange()返回的是numpy.ndarray()
                range(start, end, step)，返回一个list对象，起始值为start，终止值为end，但不含终止值，步长为step。只能创建int型list。
                arange(start, end, step)，与range()类似，但是返回一个array对象。需要引入import numpy as np，并且arange可以使用float型数据。

                2、range()不支持步长为小数，np.arange()支持步长为小数

                3、两者都可用于迭代
                range尽可用于迭代，而np.nrange作用远不止于此，它是一个序列，可被当做向量使用。
                '''
                for j in np.arange(BATCH_SIZE):
                    # np.arange()函数返回一个有终点和起点的固定步长的排列
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
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
    PreWork()
