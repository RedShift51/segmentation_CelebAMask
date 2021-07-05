#import tensorflow as tf

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

model_ = tf.keras.models.load_model("unet.h5")
model_.compile()
model_.save("model")

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(\
                         precision_mode=trt.TrtPrecisionMode.FP32, \
                            max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='model',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='trt_32')


####################################
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000)
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='model', conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='trt_16')
