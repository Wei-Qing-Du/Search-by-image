
import os
import tensorflow as tf
from PIL import Image

#原始图片地址
orig_picture = r'C:\Users\Z97MX-GAMING\Desktop\data'

#生成图片地址
gen_picture = r'C:\Users\Z97MX-GAMING\Desktop\generate_data'

#需要识别的物体类型
classes = {"0","1","2","3","4","5","6","7","8","9"}


#Samples
num_samples = 1340

#Make TFRecord
#Original data > Feature > Features > Example > TFRecord
#def create_record():
writer = tf.python_io.TFRecordWriter("flower_train.tfrecords")#Make new file  name is flower_train.records
#Define Writer to write the data，tf.python_io.TfRecordWriter() write to the TFRecords
for index,name in enumerate(classes):
    class_path = orig_picture + "/"+ name +"/"#Address of images file 
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name    #Every address of images
        img = Image.open(img_path)
        img = img.resize((224,224))     #Resize
        img_raw = img.tobytes()     #Transform the inages to bytes
        #Use features to package binary number and label for the tf.train.Example
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())   #Serialize th string
writer.close()

#Read TFRecord
#TFRecord > Example > Features > Feature > Original data
def read_and_decode(filename):  #读入flower_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename]) #Make queue

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':tf.FixedLenFeature([],tf.int64),
                                           'img_raw':tf.FixedLenFeature([],tf.string)
                                       })       #Gert image's data and label
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img,tf.uint8)
    img = tf.reshape(img,[224,224,3])   #reshape 224*224 3 chanel image
    label = tf.cast(label,tf.int32)    #Change data type
    return img,label


if __name__ == '__main__':
    batch = read_and_decode('flower_train.tfrecords')

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    dict={}    

    with tf.Session() as sess:
        #init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_samples):
            example, lab =sess.run(batch)   
            img = Image.fromarray(example,'RGB')

            if dict.get(lab) == None:
                dict[lab] = 1
            else:
                dict[lab] = dict[lab] + 1
            img.save(gen_picture+'/'+str(lab) + '/' + str(dict[lab]-1)+'_samples_'+str(lab)+'.jpg')
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()