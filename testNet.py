#=============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from CNNModel.CNNModel import deep_CNN
from CNNModel.PreWork import get_files

#=======================================================================
#Get a image
def get_one_image(train):# Train is a path
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   #Choice a image randomly

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    imag = img.resize([32, 32])
    image = np.array(imag)
    return image

#--------------------------------------------------------------------
#Test image
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 10

       image = tf.cast(image_array, tf.float32)#Change data type
       image = tf.image.per_image_standardization(image)#Limit pixel from the image
       #print(str(image))
       image = tf.reshape(image, [1, 32, 32, 3])

       logit = deep_CNN(image,BATCH_SIZE,N_CLASSES)#Model net

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[32, 32, 3])


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
           if max_index >= 0:
               print('This is a roses with possibility %.6f' %prediction[:, max_index])
            """
           elif max_index==0:
               print('This is a sunflowers with possibility %.6f' %prediction[:, 1])
           elif max_index==2:
               print('This is a poodle with possibility %.6f' %prediction[:, 2])
           else:
               print('This is a qiutian with possibility %.6f' %prediction[:, 3])
            """
#------------------------------------------------------------------------

if __name__ == '__main__':

    train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'
    train, train_label, val, val_label = get_files(train_dir, 0.3)
    img = get_one_image(val)  #通过改变参数train or val，进而验证训练集或测试集
    #print(str(img))
    evaluate_one_image(img)
#===========================================================================