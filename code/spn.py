import tensorflow as tf
from tensorflow.keras import layers

#import sys
#sys.path.append("unet/src/unet")
#from unet import *

class spnStage:
    def __init__(self, channels: int = 1, num_classes: int = 2, layer_depth: int = 5,
                filters_root: int = 64, kernel_size: int = 3, pool_size: int = 2,
                dropout_rate: int = 0.5, padding:str="valid", activation="relu"):
        self.channels = channels
        self.num_classes = num_classes
        self.layer_depth = layer_depth
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.batch_size_ = 1

    def spnForward(self, x_, w_):
        concatConst = tf.zeros((self.batch_size_, 1, 1, 1))

        # 1st dim, lr
        recurrFun1 = lambda a, i: w_[:, i:i+1, :, :1] * a + \
                    tf.concat([w_[:, i:i+1, 1:, :1], concatConst], 2) * \
                        tf.concat([a[:, :, 1:], concatConst], 2) + \
                    tf.concat([concatConst, w_[:, i:i+1, :-1, :1]], 2) * \
                        tf.concat([concatConst, a[:, :, :-1]], 2) + \
                    (1. - (w_[:, i:i+1, :, :1] + \
                        tf.concat([concatConst, w_[:, i:i+1, :-1, :1]], 2) + \
                        tf.concat([w_[:, i:i+1, 1:, :1], concatConst], 2))) \
                    * x_[:, i:i+1, :]

        x1lr = tf.scan(recurrFun1, elems=tf.range(0, w_.shape[1], 1), 
                    initializer=tf.zeros(
                            [self.batch_size_, 1, w_.shape[2], 1]))
        #x1lr = x1lr[:,:,0]
        x1lr = tf.unstack(x1lr, axis=0)
        x1lr = tf.stack(x1lr, axis=1)[:,:,0]

        # 1st dim, rl
        x1rl = tf.scan(recurrFun1, elems=tf.range(w_.shape[1]-1, -1, -1), 
                    initializer=tf.zeros(
                            [self.batch_size_, 1, w_.shape[2], 1]))
        x1rl = x1rl[:,:,0]
        x1rl = tf.unstack(x1rl, axis=0)
        x1rl = tf.stack(x1rl, axis=1)

        # 2nd dim, lr
        recurrFun2 = lambda a, i: w_[:, :, i:i+1, :1] * a + \
                    tf.concat([w_[:, 1:, i:i+1, :1], concatConst], 1) * \
                        tf.concat([a[:, 1:, :], concatConst], 1) + \
                    tf.concat([concatConst, w_[:, :-1, i:i+1, :1]], 1) * \
                        tf.concat([concatConst, a[:, :-1, :]], 1) + \
                    (1. - (w_[:, :, i:i+1, :1] + \
                        tf.concat([concatConst, w_[:, :-1, i:i+1, :1]], 1) + \
                        tf.concat([w_[:, 1:, i:i+1, :1], concatConst], 1))) \
                    * x_[:, :, i:i+1]

        x2lr = tf.scan(recurrFun2, elems=tf.range(0, w_.shape[2], 1), 
                    initializer=tf.zeros(
                            [self.batch_size_, w_.shape[1], 1, 1]))
        #x2lr = x1lr[:,:,:,0]
        x2lr = tf.unstack(x2lr, axis=0)
        x2lr = tf.stack(x2lr, axis=2)[:,:,:,:,0]

        # 2nd dim, rl
        x2rl = tf.scan(recurrFun2, elems=tf.range(w_.shape[2]-1, -1, -1), 
                    initializer=tf.zeros(
                            [self.batch_size_, w_.shape[1], 1, 1]))
        #x2rl = x1rl[:,:,:,0]
        x2rl = tf.unstack(x2rl, axis=0)
        x2rl = tf.stack(x2rl, axis=2)[:,:,:,:,0]

        x1rl = tf.concat([x1rl, x1lr, x2rl, x2lr], axis=-1)
        x1rl = tf.reduce_max(x1rl, axis=-1, keepdims=True)
        return tf.sigmoid(x1rl)


    def __call__(self, x_pair):
        x, w = x_pair
        x = self.spnForward(x, w)
        return x
