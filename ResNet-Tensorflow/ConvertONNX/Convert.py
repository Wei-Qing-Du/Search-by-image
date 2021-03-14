import sys
import win_unicode_console           
win_unicode_console.enable()          #
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx                   
import onnx                          
from tensorflow.python.keras.models import load_model

inputpath = sys.argv[1] #"../model-resnet50-final.h5"
outputpath = sys.argv[2] #'../model.onnx'

model = load_model(inputpath)
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, outputpath)