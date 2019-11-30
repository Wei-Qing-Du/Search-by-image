'''
对搭建好的网络进行训练，并保存训练参数，以便下次使用
'''
import os
import numpy as np
import tensorflow as tf
from CNN.PreWork import get_files,get_batch
from CNN.CNNModel import deep_CNN,losses,training,evaluation



N_CLASSES = 10   #images classes
#IMG_W = 28  # resize image
IMG_W = 32
#IMG_H = 28
IMG_H = 32
BATCH_SIZE = 20 #how many images put per batch
CAPACITY = 200  #size of queue
MAX_STEP = 1000    
#MAX_STEP = 200
learning_rate = 0.0001  #

#get batch
train_dir = r'C:\Users\Z97MX-GAMING\Desktop\train'   #training samples path

logs_train_dir = r'C:\Users\Z97MX-GAMING\Desktop\train_log'  #logs path
#logs_test_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data'

train,train_label,val,val_label = get_files(train_dir,0.2)      #20% validation set


#training data and labels
train_batch,train_label_batch = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

#validation data and labels
val_batch,val_label_batch = get_batch(val,val_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

#About training
train_logits = deep_CNN(train_batch,BATCH_SIZE,N_CLASSES)
train_loss = losses(train_logits,train_label_batch)
train_op = training(train_loss,learning_rate)
train_acc = evaluation(train_logits,train_label_batch)


#About testing
test_logits = deep_CNN(val_batch,BATCH_SIZE,N_CLASSES)
test_loss = losses(test_logits,val_label_batch)
test_op = training(test_loss,learning_rate)
test_acc = evaluation(test_logits,val_label_batch)

#Log report
summary_op = tf.summary.merge_all()#About visualization


sess = tf.Session()
#Make a writer to write log
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
#val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
saver = tf.train.Saver()

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
#Training each batches
try:
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break

        _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
        #_, test_loss, test_acc = sess.run([test_op, test_loss, test_acc])

        #Print the loss and accuracy per 50 steps; meanwhile, write the log to the writer
        if step % 10 == 0:
            print('step %d,train loss = %.2f,train accuracy = %.2f%%' %(step,tra_loss,tra_acc*100))
            #print('step %d,test loss = %.2f,test accuracy = %.2f%%' % (step, test_loss, test_acc * 100))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str,step)
        #Store the model per 100 steps
        if (step+1) == MAX_STEP:
            checkpoint_path =os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()

coord.join(threads)
sess.close()