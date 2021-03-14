from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import datetime
import tensorflow as tf

DATASET_PATH  = 'D:\\Search-by-image\\data_set\\cifar_10'

IMAGE_SIZE = (32, 32)

INPUT_SHAPE = (32, 32, 3)

NUM_CLASSES = 10

BATCH_SIZE = 64

FREEZE_LAYERS = 2

NUM_EPOCHS = 150

WEIGHTS_FINAL = 'model-resnet50-final.h5'

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.isdir(".\\logs"):
    os.mkdir(".\\logs")
    
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=0,
    write_images=True,
    update_freq="epoch")

callbacks = [
    tensorboard
]
assert(os.path.isdir(DATASET_PATH))

# 透過 data augmentation to make train and test data
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train1',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/test1',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# Output class index
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=INPUT_SHAPE,  classes=NUM_CLASSES)
x = net.output
x = Flatten()(x)

# add DropOut layer
x = Dropout(0.5)(x)

#Fully connected layer 1
fc1 = tf.keras.layers.Dense(100, activation='relu', name="AddedDense1")(x)

# Use softmax
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(fc1)

net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# Use Adam optimizer
net_final.compile(optimizer=Adam(lr=0.00001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Whole network
print(net_final.summary())

# Training model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks = callbacks)

net_final.save(WEIGHTS_FINAL)