import os
import tensorflow as tf
import sys
import numpy as np
from model import *

import argparse
from onnx2trt.create_engine import *
#import matplotlib.pyplot as plt


def parse_args():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--mode", type=str, default="convert")

    # train
    parser_.add_argument("--path", type=str, default="/Train_Data")
    parser_.add_argument("--epochs", type=int, default=50)
    parser_.add_argument("--logdir", type=str, default="summary")
    parser_.add_argument("--nn_type", type=str, default="unet_weak")

    # convert
    parser_.add_argument("--tfmodel_path", type=str, default="model")
    parser_.add_argument("--onnx_path", type=str, default="unet.onnx")
    parser_.add_argument("--batch_size", type=int, default=1)
    parser_.add_argument("--engine_name", type=str, default="unet.plan")

    args_ = parser_.parse_args()

    return vars(args_)

def train(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    network = unet(path_data=args["path"], epochs=args["epochs"], logdir=args["logdir"])
    network.build_nn(args["nn_type"])
    network.train()

def convert(args):
    #tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.keras.backend.set_learning_phase(0)

    batch_input_shape = (None, 256, 256, 3)
    network = unet(path_data=args["path"], epochs=args["epochs"], logdir=args["logdir"])
    network.build_nn(args["nn_type"])

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.keras.backend.set_learning_phase(0)

    # tf to onnx
    command = "python3 -m tf2onnx.convert --saved-model " + args["tfmodel_path"] + " " + \
               "--output " + args["onnx_path"]
    os.system(command)

    # onnx to tensorrt

    serialize_and_save(network.model_, args["batch_size"], 
            batch_input_shape, args["onnx_path"], args["engine_name"]) 


if __name__ == "__main__":
    args_ = parse_args()

    if args_["mode"] == "train":
        train(args_)
    elif args_["mode"] == "convert":
        convert(args_)
