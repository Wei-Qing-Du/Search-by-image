#=============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from CNNModel.CNNModel import deep_CNN
from CNNModel.PreWork import get_files

#=======================================================================
#获取一张图片
def get_one_image(train):
    #输入参数：train,训练图片的路径
    #返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   #随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    imag = img.resize([256, 256])  #由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image

#--------------------------------------------------------------------
#测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       #print(str(image))
       image = tf.reshape(image, [1, 256, 256, 3])

       logit = deep_CNN(image,BATCH_SIZE,N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[256, 256, 3])


       # you need to change the directories to yours.
       logs_train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'

       saver = tf.train.Saver()

       with tf.Session() as sess:
           tf.global_variables_initializer().run()

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               print(global_step)
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a roses with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a sunflowers with possibility %.6f' %prediction[:, 1])
           #elif max_index==2:
           #    print('This is a poodle with possibility %.6f' %prediction[:, 2])
           #else:
           #    print('This is a qiutian with possibility %.6f' %prediction[:, 3])

#------------------------------------------------------------------------

if __name__ == '__main__':

    train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'
    train, train_label, val, val_label = get_files(train_dir, 0.3)
    img = get_one_image(val)  #通过改变参数train or val，进而验证训练集或测试集
    #print(str(img))
    evaluate_one_image(img)
#===========================================================================