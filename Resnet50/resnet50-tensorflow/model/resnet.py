import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd() + '\\Resnet50\\resnet50-tensorflow\\data')
from model.layers import *
from data_Resnet50 import divide_set, get_images_from_folder, parse_json

from tensorflow.python import debug as tf_debug

import logging
import time


class ResNet50(object):
    '''
    ResNet 50 Model
    '''

    def __init__(self, config, n_classes):
        # Config Logging and log TF Version
        logging.basicConfig(level=logging.INFO)
        logging.info("-----------------------------------------")
        logging.info("         USING TF Version {}".format(tf.__version__))

        self.is_training = True

        # load hyperparameters with config file
        # (image), (optimizer), batch_size
        (image_size, n_channels, n_classes), (lr, beta1, beta2, epsilon), batch_size = parse_json(config)

        # image
        self.image_size = image_size
        self.n_channels = n_channels

        self.x, self.y,self. y_pred_cls, self.global_step, self.learning_rate = 0

        # For optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # batch size
        self.batch_size = batch_size

        # Output Layer
        self.num_classes = n_classes

        # Input Layer
        self.input_tensor = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.n_channels])

        logging.info("---------------------------------------")
        logging.info("             MODEL CONFIG              ")
        logging.info("               OPTIMIZER               ")
        logging.info(" LR: {}, BETA1: {}, BETA2: {}, EPSILON: {}".format(self.lr, self.beta1, self.beta2, self.epsilon))
        logging.info("                 IMAGE                 ")
        logging.info("  IMAGE SIZE: {}, CHANNELS_NUMBER: {}  ".format(self.image_size, self.n_channels))
        logging.info("                TRAINING               ")
        logging.info("             BATCH SIZE: {}            ".format(self.batch_size))
       
    # ----------------------------  GRAPH  ----------------------------- #


    def inference(self):
        '''
        Defining model's graph
        '''
        with tf.name_scope('main_params'):
            x = tf.placeholder(tf.float32, shape=[None, self.image_size * self.image_size * self.n_channels], name='Input')
            y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
            x_image = tf.reshape(x, [-1, self.image_size * self.image_size, self.n_channels], name='images')
            global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

            # Stage 1
            self.conv1 = conv_layer(x_image, 7, self.n_channels, 64, 2, 'scale1')
            self.conv1 = bn(self.conv1, self.is_training, 'scale1')
            self.conv1 = relu(self.conv1)
            self.pool1 = maxpool(self.conv1, name='pool1')

        """
        Every stages have different number of blocks that can reference from https://zhuanlan.zhihu.com/p/79378841
        Every blocks have three layers.

        End of layer to next layer(begin of next stage) that will use short cut.
        """
        # Stage 2
        with tf.variable_scope('scale2'):
            self.block1_1 = res_block_3_layer(self.pool1, [64, 64, 256], 'block1_1', True, 1, self.is_training)
            self.block1_2 = res_block_3_layer(self.block1_1, [64, 64, 256], 'block1_2', False, 1, self.is_training)
            self.block1_3 = res_block_3_layer(self.block1_2, [64, 64, 256], 'block1_3', False, 1, self.is_training)

        # Stage 3
        with tf.variable_scope('scale3'):
            self.block2_1 = res_block_3_layer(self.block1_3, [128, 128, 512], 'block2_1', True, 2, self.is_training)
            self.block2_2 = res_block_3_layer(self.block2_1, [128, 128, 512], 'block2_2', False, 1, self.is_training)
            self.block2_3 = res_block_3_layer(self.block2_2, [128, 128, 512], 'block2_3', False, 1, self.is_training)
            self.block2_4 = res_block_3_layer(self.block2_3, [128, 128, 512], 'block2_4', False, 1, self.is_training)

        # Stage 4
        with tf.variable_scope('scale4'):
            self.block3_1 = res_block_3_layer(self.block2_4, [256, 256, 1024], 'block3_1', True, 2, self.is_training)
            self.block3_2 = res_block_3_layer(self.block3_1, [256, 256, 1024], 'block3_2', False, 1, self.is_training)
            self.block3_3 = res_block_3_layer(self.block3_2, [256, 256, 1024], 'block3_3', False, 1, self.is_training)
            self.block3_4 = res_block_3_layer(self.block3_3, [256, 256, 1024], 'block3_4', False, 1, self.is_training)
            self.block3_5 = res_block_3_layer(self.block3_4, [256, 256, 1024], 'block3_5', False, 1, self.is_training)
            self.block3_6 = res_block_3_layer(self.block3_5, [256, 256, 1024], 'block3_6', False, 1, self.is_training)

        # Stage 5
        with tf.variable_scope('scale5'):
            self.block4_1 = res_block_3_layer(self.block3_6, [512, 512, 2048], 'block4_1', True, 2, self.is_training)
            self.block4_2 = res_block_3_layer(self.block4_1, [512, 512, 2048], 'block4_2', False, 1, self.is_training)
            self.block4_3 = res_block_3_layer(self.block4_2, [512, 512, 2048], 'block4_3', False, 1, self.is_training)

        # Fully-Connected
        with tf.variable_scope('fc'):
            self.pool2 = avgpool(self.block4_3, 7, 1, 'pool2')
            self.logits = fc_layer(self.pool2, 2048, self.num_classes, 'fc1')
            drop = tf.layers.dropout(fc, rate=0.5)
            softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, name=scope.name)

            y_pred_cls = tf.argmax(softmax, axis=1, name ="predicted_labels")

        return x, y, softmax, y_pred_cls, global_step, learning_rate


    # ---------------------------  TRAINING  ---------------------------- #


    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.y)
            self.loss = tf.reduce_mean(entropy, name='loss')


    def optimize(self):
        self.global_step = tf.train.get_or_create_global_step()# The number of batches seen by the graph
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            name='Adam'
        ).minimize(self.loss, global_step=self.global_step)


    def average_scalars(self):
        self.avg_loss = tf.Variable(0.0)
        self.avg_acc = tf.Variable(0.0)


    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            predictions = softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


    def summary(self):
        '''
        Logging to TensorBoard
        '''
        with tf.name_scope('batch-summaries'):
            tf.summary.scalar('batch-loss', self.loss)
            tf.summary.scalar('batch-accuracy', self.accuracy)
            tf.summary.histogram('histogram-loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        with tf.name_scope('trainig'):
            loss_summary = tf.summary.scalar('loss', self.avg_loss)
            acc_summary = tf.summary.scalar('acc', self.avg_acc)
            self.avg_summary_op = tf.summary.merge([loss_summary, acc_summary])


    def build(self):
        '''
        Build computational graph
        '''
        self.x, self.y, self.labels, self.y_pred_cls, self.global_step, self.learning_rate =self.inference()
        self.loss()
        self.optimize()
        self.average_scalars()
        self.eval()
        self.summary()


    def write_average_summary(self, sess, writer, epoch, avg_loss, avg_acc):
        summaries = sess.run(self.avg_summary_op, {self.avg_loss: avg_loss, self.avg_acc: avg_acc})
        writer.add_summary(summaries, global_step=epoch)
        writer.flush()

    def test_and_save(_global_step, epoch):
        global global_accuracy
        global epoch_start

        i = 0
        predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
        while i < len(test_x):
            j = min(i + _BATCH_SIZE, len(test_x))
            batch_xs = test_x[i:j, :]
            batch_ys = test_y[i:j, :]
            predicted_class[i:j] = sess.run(
                y_pred_cls,
                feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
            )
            i = j

        correct = (np.argmax(test_y, axis=1) == predicted_class)
        acc = correct.mean()*100
        correct_numbers = correct.sum()

        hours, rem = divmod(time() - epoch_start, 3600)
        minutes, seconds = divmod(rem, 60)
        mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
        print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

        if global_accuracy != 0 and global_accuracy < acc:

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            ])
            train_writer.add_summary(summary, _global_step)

            saver.save(sess, save_path=_SAVE_PATH_OF_CKPT, global_step=_global_step)

            tf.train.write_graph(sess.graph_def, '.', 'minimal_graph.proto', as_text=False)

            mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
            print(mes.format(acc, global_accuracy))
            global_accuracy = acc

        elif global_accuracy == 0:
            global_accuracy = acc

        print("###########################################################################################################")

    def train_one_epoch(self, sess, saver, init, writer, epoch):
        start_time = time.time()
        for s in range(self.batch_size):
            batch_xs = train_x[s*self.batch_size: (s+1)*self.batch_size]
            batch_ys = train_y[s*self.batch_size: (s+1)*self.batch_size]
            sess.run(init)
            self.training = True
            total_loss = 0
            total_acc = 0
            n_batches = 0
            try:
                while True:
                    i_global, loss_batch, acc_batch, summaries = sess.run([global_step, self.optimizer, self.loss, self.accuracy, self.summary_op],
                        feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)})
                    writer.add_summary(summaries, global_step=step)
                    total_loss += loss_batch
                    total_acc += acc_batch
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass
        
        if s % 10 == 0:
            avg_loss = total_loss/n_batches
            avg_acc = total_acc/n_batches/self.batch_size
            self.write_average_summary(sess, writer, epoch, avg_loss, avg_acc)
            logging.info('Training loss at epoch {0}: {1}'.format(epoch, avg_loss))
            logging.info('Training accuracy at epoch {0}: {1}'.format(epoch, avg_acc))
            logging.info('Took: {0} seconds'.format(time.time() - start_time))

        test_and_save(i_global, epoch)

    def train(self, n_epochs, debug=False):
        '''
        This train function alternates between training and evaluating once per epoch run
        '''
        # Config Logging
        logging.basicConfig(level=logging.INFO)

        train_writer = tf.summary.FileWriter('logs/train')
        val_writer = tf.summary.FileWriter('logs/val')

        train_writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            # Wrap Debug Session
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            #sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100) #Save no.100 model recently.
            # upload existing saves
            #train_step = self.global_step.eval()
            #val_step = train_step
            for epoch in range(n_epochs):
                train_step = self.train_one_epoch(sess, saver, self.train_iterator_init_op, train_writer, epoch)
                #val_step = self.eval_once(sess, self.train_iterator_init_op, val_writer, epoch, val_step)
                # Save Each Epoch
                save_path = saver.save(sess, "training/epoch{}/model.ckpt".format(epoch))
        writer.close()


    # ---------------------------  PREDICTION  ---------------------------- #


    def load_pred(self, folder):
        '''
        Creates Prediction Dataset
        '''
        # Get All Images From Folder
        filenames = get_images_from_folder(folder)
        # Calculating Batch Size
        batch = len(filenames)
        logging.info("       FOUND {0} IMAGES TO PREDICT".format(batch))

        # Parsing Images
        parse_fn = lambda f: _parse_image(f, self.n_channels, self.image_size)

        # Creates Iterator over the Prediction DataSet
        with tf.name_scope('predict-data'):
            predict_dataset = (tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
                .map(parse_fn, num_parallel_calls=4)
                .batch(batch)
            )

            predict_iterator = predict_dataset.make_initializable_iterator()
            self.predict_iterator_init_op = predict_iterator.initializer

        # Pass Batch of Images into Model's Input
        self.images = predict_iterator.get_next()
        # Return filenames to match them with Predictions
        return filenames


    def predict(self, weights, is_logging=False, debug=False):
        '''
        Loads graph and weights,
        creates a feed dict and passes it through the model
        returns predictions
        '''
        # Build Computational Graph
        self.inference()
        self.training = False

        # Logging to TensorBoard
        if is_logging:
            writer = tf.summary.FileWriter('logs/predict')
            writer.add_graph(tf.get_default_graph())
            writer.close()

        # Get Predictions
        with tf.Session() as sess:
            # Wrap Debug Session
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # Restoring Weights From Trained Model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, weights)
            logging.info("Finished restoring weights")

            # Passing images through Model
            start_time = time.time()
            logging.info("Started evaluating images")
            sess.run(self.predict_iterator_init_op)
            predictions = sess.run(softmax(self.logits))
            logging.info('Took: {0} seconds'.format(time.time() - start_time))

        # Return Prediction for further Results Interpretation
        return predictions

    def lr(epoch):
        learning_rate = 1e-3
        if epoch > 80:
            learning_rate *= 0.5e-3
        elif epoch > 60:
            learning_rate *= 1e-3
        elif epoch > 40:
            learning_rate *= 1e-2
        elif epoch > 20:
            learning_rate *= 1e-1
        return learning_rate