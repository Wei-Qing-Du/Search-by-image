import tensorflow as tf

'''
    该网络结构包括：
        卷积池化层：3
        全连接层：2
        激活函数：ReLU
        Dropout、分类器；
'''
'''
在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。
 
想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer
tf.placehold与tf.Variable的区别：
    tf.placehold 占位符
        主要为真实输入数据和输出标签的输入， 用于在 feed_dict中的变量，不需要指定初始值，具体值在feed_dict中的变量给出。
    tf.Variable 主要用于定义weights bias等可训练会改变的变量，必须指定初始值。
        通过Variable()构造函数后，此variable的类型和形状固定不能修改了，但值可以用assign方法修改。
 
tf.get_variable和tf.Variable函数差别
相同点：通过两函数创建变量的过程基本一样，
        tf.variable函数调用时提供的维度(shape)信息以及初始化方法(initializer)的参数和tf.Variable函数调用时提供的初始化过程中的参数基本类似。
不同点：两函数指定变量名称的参数不同，
        对于tf.Variable函数，变量名称是一个可选的参数，通过name="v"的形式给出
        tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量
'''

#函数申明
def weight_variable(shape,n):
    # tf.truncated_normal(shape, mean, stddev)这个函数产生正态分布，均值和标准差自己设定。
    # shape表示生成张量的维度，mean是均值
    # stddev是标准差,，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape,stddev=n,dtype=tf.float32)
    return initial

def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1,shape=shape,dtype=tf.float32)
    return initial

def conv2d(x,w):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    # padding 一般只有两个值
    # 卷积层后输出图像大小为：（W+2P-f）/stride+1并向下取整
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')   #[batch, height, width, channels]
    #strides[0] = 1，也即在 batch 维度上的移动为 1，也就是不跳过任何一个样本，否则当初也不该把它们作为输入（input）
    #strides[3] = 1，也即在 channels 维度上的移动为 1，也就是不跳过任何一个颜色通道；

def max_pooling(x,name):    #2×2
    # 池化卷积结果（conv2d）池化层采用kernel大小为3*3，步数也为2，SAME：周围补0，取最大值。数据量缩小了4倍
    # x 是 CNN 第一步卷积的输出量，其shape必须为[batch, height, weight, channels];
    # ksize 是池化窗口的大小， shape为[batch, height, weight, channels]
    # stride 步长，一般是[1，stride， stride，1]
    # 池化层输出图像的大小为(W-f)/stride+1，向上取整
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name=name)

#A samole CNN，conv and pool layers ×2，full connect layer ×2，use softmax to classified
#64 3×3 conv layers（3 channels），padding='SAME'，size of image is same ad orignal one after conv
def deep_CNN(images,batch_size,n_classes):
    with tf.variable_scope('conv1') as scope:
        #Frist conv layer 
        w_conv1 = tf.Variable(weight_variable([3,3,3,64],1.0),name='weights',dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]),name='biases',dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images,w_conv1)+b_conv1,name='conv1')   #weight * x(num of conv) + bias

    #Frist max pool layer
    #3*3 pool layer，strides is 2。
    #tf.nn.lrn is Local Response Normalization
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pooling(h_conv1,'pooling1')     #128*128*64
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    #Second conv layer
    #32 3×3 conv layers（3 channels），padding='SAME'，size of image is same ad orignal one after conv
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3,3,64,32],0.1),name='weights',dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]),name='biases',dtype=tf.float32)   #32个偏置值
        h_conv2 = tf.nn.relu(conv2d(norm1,w_conv2)+b_conv2,name='conv2')

    #Second max pool layer
    #3*3 pool layer，strides is 2。
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pooling(h_conv2,'pooling2')
        norm2 = tf.nn.lrn(pool2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')

    #第三层卷积层
    #16个3*3卷积核（16个通道），padding='SAME'，表示padding后卷积的图与原图尺寸一致，激活函数relu
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3,3,32,16],0.1),name='weights',dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]),name='biases',dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2,w_conv3)+b_conv3,name='conv3')

    #第三层池化层
    ##3*3最大池化，步长strides为2,池化后执行lrn()操作
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pooling(h_conv3,'pooling3')
        norm3 = tf.nn.lrn(pool3,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm3')

#   第四层卷积层（后加的）
    with tf.variable_scope('conv4') as scope:
        w_conv4 = tf.Variable(weight_variable([3,3,16,8],0.1),name='weights',dtype=tf.float32)
        b_conv4 = tf.Variable(bias_variable([8]),name='biases',dtype=tf.float32)
        h_conv4 = tf.nn.relu(conv2d(norm3,w_conv4)+b_conv4,name='conv4')

    with tf.variable_scope('pooling4_lrn') as scope:
        pool4 = max_pooling(h_conv4,'pooling4')
        norm4 = tf.nn.lrn(pool4,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm4')

    #第四层，全连接层-1
    #256个神经元，将之前pool层的输出reshape成一行，激活函数relu（）
    with tf.variable_scope('local5') as scope:
        reshape = tf.reshape(norm4,shape=[batch_size,-1])
        dim = reshape.get_shape()[1].value
#        w_fc1 = tf.Variable(weight_variable([dim,256],0.005), name='weights',dtype=tf.float32)
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[dim,256],stddev=0.005,dtype=tf.float32),
                            name='weights',dtype=tf.float32)
#        b_fc1 = tf.Variable(bias_variable([256]),name='biases',dtype=tf.float32)
        b_fc1 = tf.Variable(tf.constant(value=0.1,dtype=tf.float32,shape=[256]),
                            name='biases',dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape,w_fc1)+b_fc1,name=scope.name)

    #第五层，全连接层-2
    #256个神经元，激活函数relu()
    with tf.variable_scope('local6') as scope:
#        w_fc2 = tf.Variable(weight_variable([256,256],0.005),name='weights',dtype=tf.float32)
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[256,256],stddev=0.005,dtype=tf.float32),
                            name='weights',dtype=tf.float32)
#        b_fc2 = tf.Variable(bias_variable([256]),name='biases',dtype=scope.name)
        b_fc2 = tf.Variable(tf.constant(value=0.1,dtype=tf.float32,shape=[256]),
                            name='biases',dtype=tf.float32)
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2,name=scope.name)

    #对卷积结果进行Dropout操作
    h_fc2_dropout = tf.nn.dropout(h_fc2,0.5)

    #softmax回归层
    with tf.variable_scope('softmax_liner') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[256,n_classes],stddev=0.005,dtype=tf.float32)
                              ,name='softmax_liner',dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1,dtype=tf.float32,shape=[n_classes])
                             ,name='biases',dtype=tf.float32)
        softmax_liner = tf.add(tf.matmul(h_fc2_dropout,weights),biases,name='softmax_liner')
    return softmax_liner

#loss计算
    #传入参数：logits，网络计算输出值。labels，真实值，在这里是0或1
    #返回参数：loss，损失值
def losses(logits,lablels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=lablels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

#loss损失值优化

def training(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

#评价计算/准确率计算
def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy