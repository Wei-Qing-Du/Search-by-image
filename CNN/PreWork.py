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

train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'

roses = []
label_roses = []
sunflowers = []
label_sunflowers = []

#step1:获取Image_to_tfrecords.py文件运行生成后的图片路径
    #获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'/roses'):
        roses.append(file_dir+'/roses'+'/'+file)
        label_roses.append(0)
    for file in os.listdir(file_dir+'/sunflowers'):
        sunflowers.append(file_dir+'/sunflowers'+'/'+file)
        label_sunflowers.append(1)

    #打印出提取图片的情况，检测是否正确提取
    print("There are %d roses\nThere are %d sunflowers\n"%(len(roses),len(sunflowers)),end="")

#step2: 对生成图片路径和标签list做打乱处理把roses和sunflowers合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），
    # 函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((roses,sunflowers))
    label_list = np.hstack((label_roses,label_sunflowers))

    #利用shuffle,转置，随机打乱
    temp = np.array([image_list,label_list])    #转换成2维矩阵
    temp = temp.transpose()     #转置
    np.random.shuffle(temp)     #按行随机打乱顺序函数
    #print(temp)
    #从打乱的temp中再取出list（img和lab）
    #image_list = list(temp[:,0])
    #label_list = list(temp[:,1])
    #label_list = [int(i) for i in label_list]
    #return  image_list,label_list

    #将所有的img和lab转换成list
    all_image_list = list(temp[:,0])    #取出第0列数据，即图片路径
    all_label_list = list(temp[:,1])    #取出第1列数据，即图片标签

    #将所得list分为两部分，一部分用来train，一部分用来测试val
    #ratio是测试集比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  #测试样本数
    n_train = n_sample - n_val    #训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
#    tra_labels = [int(float(i)) for i in tra_labels]    #转换成int数据类型
    tra_labels = [int(i) for i in tra_labels]

    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
#    val_labels = [int(float(i)) for i in val_labels]    #转换成int数据类型
    val_labels = [int(i) for i in val_labels]
    #print(val_labels)

    return tra_images,tra_labels,val_images,val_labels

#--------------------------生成batch------------------------

#step1：将上面生成的list传入get_batch（），转换类型，产生一个输入队列queue，因为img和lab是分开的
#所以使用tf.train.slice_input_producer(),然后用tf.read_file()从队列中读取图像
#   image_w，image_H ：设置好固定的图像高度和宽度
#   设置batch_size : 每个batch要放多少张图片
#   capacity：一个队列最大多少

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #用tf.cast()转换类型
    image = tf.cast(image,tf.string)    #可变长度的字节数组，每一个张量元素都是一个字节数组
    label = tf.cast(label,tf.int32)     #

#    print(label)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    #make an input queue
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像

    #step2:将图像解码，使用相同类型的图像。不同类型的图像不能混合在一起，要么只用JPEG，要么只用PNG等
#    image = tf.image.decode_image(image_contents,channels=3)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    #step3:数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的图形更健壮
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)   #对resize后的图片进行标准化处理

#    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉105行（标准化）和 121行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
#    image = tf.image.per_image_standardization(image)
#具体解释地址看该链接代码：https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/input_data.py

    #step4:生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch ,label_batch = tf.train.batch([image,label],
                                              batch_size = batch_size,
                                              num_threads= 32,
                                              capacity =capacity)
    #重新排列label，行数为[batch_size]
    label_batch =tf.reshape(label_batch,[batch_size])
#    image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image_batch,tf.float32)     #显示灰度图像

    return image_batch,label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据


"""
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
"""