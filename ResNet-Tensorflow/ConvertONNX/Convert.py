import win_unicode_console           
win_unicode_console.enable()          #
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx                   
import onnx                          
from tensorflow.python.keras.models import load_model

model = load_model("../model-resnet50-final.h5")
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = '../model.onnx'
onnx.save_model(onnx_model, temp_model_file)