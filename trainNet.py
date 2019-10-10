'''
对搭建好的网络进行训练，并保存训练参数，以便下次使用
'''
import os
import numpy as np
import tensorflow as tf
from CNN.PreWork import get_files,get_batch
from CNN.CNNModel import deep_CNN,losses,training,evaluation


#变量申明
N_CLASSES = 2   #roses,sunflowers
#IMG_W = 28  # resize图像，太大的话训练时间久
IMG_W = 32
#IMG_H = 28
IMG_H = 32
BATCH_SIZE = 20 #每个batch要放多少张图片
CAPACITY = 200  #一个队列最大多少
MAX_STEP = 1000    #一般大于10K
#MAX_STEP = 200
learning_rate = 0.0001  #

#获取批次batch
train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'   #训练样本的读入路径

logs_train_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data/1'  #logs存储路径
#logs_test_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'

train,train_label,val,val_label = get_files(train_dir,0.2)      #验证集比例20%


#训练数据及标签
train_batch,train_label_batch = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

#测试数据及标签
val_batch,val_label_batch = get_batch(val,val_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

#训练操作定义
train_logits = deep_CNN(train_batch,BATCH_SIZE,N_CLASSES)
train_loss = losses(train_logits,train_label_batch)
train_op = training(train_loss,learning_rate)
train_acc = evaluation(train_logits,train_label_batch)


#测试操作定义
test_logits = deep_CNN(val_batch,BATCH_SIZE,N_CLASSES)
test_loss = losses(test_logits,val_label_batch)
test_op = training(test_loss,learning_rate)
test_acc = evaluation(test_logits,val_label_batch)

#这个是log汇总记录
summary_op = tf.summary.merge_all()

#产生一个会话
sess = tf.Session()
#产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
#val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
saver = tf.train.Saver()
#所有节点初始化
sess.run(tf.global_variables_initializer())
#队列监控
coord =tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
'''
for step in range(MAX_STEP):
    _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
    if step % 10 == 0:
        print('step %d ,train loss = %.2f,train accuracy = %.2%%' %(step,tra_loss,tra_acc*100))
        summary_str = sess.run(summary_op)
        train_writer.add_summary(summary_str,step)

    if (step+1) == MAX_STEP:
        checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
        saver.save(sess,checkpoint_path,global_step=step)
'''
'''
'''
#进行batch的训练
try:
    #执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        #启动以下操作节点，

        _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
        #_, test_loss, test_acc = sess.run([test_op, test_loss, test_acc])

        #每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 10 == 0:
            print('step %d,train loss = %.2f,train accuracy = %.2f%%' %(step,tra_loss,tra_acc*100))
            #print('step %d,test loss = %.2f,test accuracy = %.2f%%' % (step, test_loss, test_acc * 100))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str,step)
        #每隔100步，保存一次训练好的模型
        if (step+1) == MAX_STEP:
            checkpoint_path =os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()

coord.join(threads)    # 把开启的线程加入主线程，等待threads结束
sess.close()