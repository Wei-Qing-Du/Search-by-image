# Search image by deep learning

We try to use deep learning Deep Convolutional Neural Networks to catch image features which will be a classifier to let computer know image type and then find similar images from local computer.

# References
Our project referenced **Image Retrieval[1]** and **Tensorflow-cifar-10[2]** to training the data.
>* [1]CH Kuo, YH Chou, PC Chang, "Using Deep Convolutional Neural Networks for Image Retrieval", Electronic Imaging, 2016.
>* [2][Tensorflow-cifar-10.](https://github.com/exelban/tensorflow-cifar-10)
# Environment
>* WINDOWS 10.
>* Python3.x.
>* Anaconda.
>* VisualStudio 2017 up.
>* Tensorflow gpu.
>   * Use gpu can speed up training.
>* [Open Neural Network Exchange(ONNX)](https://github.com/onnx/onnx)  
>   * Tool for converting tensorflow to ONNX.
>* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) for training and testing.  

Cifar-10 convolutional network implementation example using TensorFlow library.
![](https://trello-attachments.s3.amazonaws.com/5e11b4a007fc4d333fd1819b/1063x532/7ffdae91082a8a57c9e9649ac90b9ee0/image.png)

# Workflow

Frist, we need to convert to ONNX model after train to tensorflow model, but not transform diametrically. It need to [freeze graph with tensorflow](https://github.com/onnx/tensorflow-onnx)(xx.ckpt) that makes .pb file.  
After convert to ONNX model we wrote [python code](predict_test.py) to run it and use C# to connect to the pyhton to recognize image type.
![](WorkFlow/WorkFlow.jpg)


# Accuracy 
Best accurancy what I receive was ```78-79%``` on test data set. We'll use **data augmentation** and other network, for example, **ResNet** in the future, 

This repository is just example of implemantation convolution neural network. Here I implement a simple neural network for image recognition with good if you want to get more that 80% accuracyaccuracy.


# Result
By default network will be run 60 epoch (60 times on all training data set).  
You can change that by editing ```_EPOCH``` in ```train.py``` file.

Also by default it process 128 files in each step.  
If you training network on CPU or GPU (lowest that 1060 6GB) change ```_BATCH_SIZE``` in ```train.py``` to a smaller value.


```sh
python3 train.py
```

Simple output:
```sh
Epoch: 60/60

Global step: 23070 - [>-----------------------------]   0% - acc: 0.9531 - loss: 1.5081 - 7045.4 sample/sec
Global step: 23080 - [>-----------------------------]   3% - acc: 0.9453 - loss: 1.5159 - 7147.6 sample/sec
Global step: 23090 - [=>----------------------------]   5% - acc: 0.9844 - loss: 1.4764 - 7154.6 sample/sec
Global step: 23100 - [==>---------------------------]   8% - acc: 0.9297 - loss: 1.5307 - 7104.4 sample/sec
Global step: 23110 - [==>---------------------------]  10% - acc: 0.9141 - loss: 1.5462 - 7091.4 sample/sec
Global step: 23120 - [===>--------------------------]  13% - acc: 0.9297 - loss: 1.5314 - 7162.9 sample/sec
Global step: 23130 - [====>-------------------------]  15% - acc: 0.9297 - loss: 1.5307 - 7174.8 sample/sec
Global step: 23140 - [=====>------------------------]  18% - acc: 0.9375 - loss: 1.5231 - 7140.0 sample/sec
Global step: 23150 - [=====>------------------------]  20% - acc: 0.9297 - loss: 1.5301 - 7152.8 sample/sec
Global step: 23160 - [======>-----------------------]  23% - acc: 0.9531 - loss: 1.5080 - 7112.3 sample/sec
Global step: 23170 - [=======>----------------------]  26% - acc: 0.9609 - loss: 1.5000 - 7154.0 sample/sec
Global step: 23180 - [========>---------------------]  28% - acc: 0.9531 - loss: 1.5074 - 6862.2 sample/sec
Global step: 23190 - [========>---------------------]  31% - acc: 0.9609 - loss: 1.4993 - 7134.5 sample/sec
Global step: 23200 - [=========>--------------------]  33% - acc: 0.9609 - loss: 1.4995 - 7166.0 sample/sec
Global step: 23210 - [==========>-------------------]  36% - acc: 0.9375 - loss: 1.5231 - 7116.7 sample/sec
Global step: 23220 - [===========>------------------]  38% - acc: 0.9453 - loss: 1.5153 - 7134.1 sample/sec
Global step: 23230 - [===========>------------------]  41% - acc: 0.9375 - loss: 1.5233 - 7074.5 sample/sec
Global step: 23240 - [============>-----------------]  43% - acc: 0.9219 - loss: 1.5387 - 7176.9 sample/sec
Global step: 23250 - [=============>----------------]  46% - acc: 0.8828 - loss: 1.5769 - 7144.1 sample/sec
Global step: 23260 - [==============>---------------]  49% - acc: 0.9219 - loss: 1.5383 - 7059.7 sample/sec
Global step: 23270 - [==============>---------------]  51% - acc: 0.8984 - loss: 1.5618 - 6638.6 sample/sec
Global step: 23280 - [===============>--------------]  54% - acc: 0.9453 - loss: 1.5151 - 7035.7 sample/sec
Global step: 23290 - [================>-------------]  56% - acc: 0.9609 - loss: 1.4996 - 7129.0 sample/sec
Global step: 23300 - [=================>------------]  59% - acc: 0.9609 - loss: 1.4997 - 7075.4 sample/sec
Global step: 23310 - [=================>------------]  61% - acc: 0.8750 - loss: 1.5842 - 7117.8 sample/sec
Global step: 23320 - [==================>-----------]  64% - acc: 0.9141 - loss: 1.5463 - 7157.2 sample/sec
Global step: 23330 - [===================>----------]  66% - acc: 0.9062 - loss: 1.5549 - 7169.3 sample/sec
Global step: 23340 - [====================>---------]  69% - acc: 0.9219 - loss: 1.5389 - 7164.4 sample/sec
Global step: 23350 - [====================>---------]  72% - acc: 0.9609 - loss: 1.5002 - 7135.4 sample/sec
Global step: 23360 - [=====================>--------]  74% - acc: 0.9766 - loss: 1.4842 - 7124.2 sample/sec
Global step: 23370 - [======================>-------]  77% - acc: 0.9375 - loss: 1.5231 - 7168.5 sample/sec
Global step: 23380 - [======================>-------]  79% - acc: 0.8906 - loss: 1.5695 - 7175.2 sample/sec
Global step: 23390 - [=======================>------]  82% - acc: 0.9375 - loss: 1.5225 - 7132.1 sample/sec
Global step: 23400 - [========================>-----]  84% - acc: 0.9844 - loss: 1.4768 - 7100.1 sample/sec
Global step: 23410 - [=========================>----]  87% - acc: 0.9766 - loss: 1.4840 - 7172.0 sample/sec
Global step: 23420 - [==========================>---]  90% - acc: 0.9062 - loss: 1.5542 - 7122.1 sample/sec
Global step: 23430 - [==========================>---]  92% - acc: 0.9297 - loss: 1.5313 - 7145.3 sample/sec
Global step: 23440 - [===========================>--]  95% - acc: 0.9297 - loss: 1.5301 - 7133.3 sample/sec
Global step: 23450 - [============================>-]  97% - acc: 0.9375 - loss: 1.5231 - 7135.7 sample/sec
Global step: 23460 - [=============================>] 100% - acc: 0.9250 - loss: 1.5362 - 10297.5 sample/sec

Epoch 60 - accuracy: 78.81% (7881/10000)
This epoch receive better accuracy: 78.81 > 78.78. Saving session...
###########################################################################################################
```


### Run network on test data set
We changed a little bit of code, let it can classify a single image to suit our project.
```sh
python3 predict.py
```
```sh
#Use one data
    batch_xs = test_x[1:2, :]
    batch_ys = test_y[1:2, :]
    for i,v in enumerate(batch_ys):
        for j ,e in enumerate(v):
         if e==1.0:
             label = j
             break
    print("real class is %d\n" %label)
    predicted_class = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
    print("predicted class is %d\n" %predicted_class)
```

Simple output:
```sh
Trying to restore last checkpoint ...
Restored checkpoint from: ./tensorboard/cifar-10-v1.0.0/-23460

Accuracy on Test-Set: 78.81% (7881 / 10000)
```

Please send me (or open issue) if you don't understand or encounter difficulties.


### v0.0
```
-  Frist Realse 
```
## License
[MIT License](https://github.com/exelban/tensorflow-cifar-10/blob/master/LICENSE)
