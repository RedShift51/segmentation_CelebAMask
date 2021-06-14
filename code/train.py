import tensorflow as tf
from tensorflow.keras import layers

import sys
#sys.path.append("unet/src/unet")
#import unet
#from datasets import *
import numpy as np
from model import *

LEARNING_RATE = 1e-3

import argparse


import matplotlib.pyplot as plt


def parse_args():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--path", type=str, default="/home/alex/Downloads/CelebAMask-HQ")
    parser_.add_argument("--epochs", type=int, default=5)
    parser_.add_argument("--logdir", type=str, default="summary")
    args_ = parser_.parse_args()

    return vars(args_)

def train():
    args = parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    """
    sd = segmGenerator(args["path"], 
                os.listdir(os.path.join(args["path"], "CelebA-HQ-img")))
    for b1, b2 in sd:
        plt.subplot(121)
        plt.imshow(b1[0])
        plt.subplot(122)
        plt.imshow(b2[0,:,:,0])
        print(np.unique(b2))
        plt.show()
        1/0
    """
    network = unet(path_data=args["path"], epochs=args["epochs"], logdir=args["logdir"])
    network.build_raw_unet()
    network.train()


if __name__ == "__main__":
    train()
