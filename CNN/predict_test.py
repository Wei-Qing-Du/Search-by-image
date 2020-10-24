# Import the ONNX runtime environment
import onnxruntime as rt
import numpy as np
from PIL import Image
import sys
import os


def preprocess(path):
    img = Image.open(path)
    img = img.resize((32, 32))

    return img

def RunTestModel(img_path):
    try:
        # setup runtime - load the persisted ONNX model
        img = preprocess(img_path)
        img.load()
    
        data = np.asarray( img, dtype="float32" )
        data = data.reshape(1, 3072)

        sess = rt.InferenceSession(sys.path[0] + "\\model.onnx")
    

        # get model metadata to enable mapping of new input to the runtime model.
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        # retrieve prediction - passing in the input list (you can also pass in multiple inputs as a list of lists)
        pred_onx = sess.run([label_name], {input_name: data})[0]

    except Exception as e:
        print(e)
    return pred_onx

print(RunTestModel(sys.argv[1]))