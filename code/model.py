import os
import tensorflow as tf
import tensorflow.io as tfio
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
import cv2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from spn import *


# batch size = 1
class segmGenerator:
    def __init__(self, path_data, images, batch_size=1):
        self.path_data_ = path_data
        self.batch_size_ = batch_size

        self.imgs_ = images#os.listdir(os.path.join(path_data, "CelebA-HQ-img"))
        self.masks_path_ = os.path.join(path_data, "CelebAMask-HQ-mask-anno")
        self.n = len(self.imgs_)
        self.i = 0

    def __len__(self):
        return len(self.imgs_)

    def __next__(self):
        if self.i == self.n:
            raise StopIteration()
        else:
            return self.__getitem__(self.i)
        self.i += 1

    def __iter__(self):
        return self

    def __call__(self):
        self.i = 0
        return self

    def __getitem__(self, idx):
        # original images have resolution 1024x1024, and I resized them by 256x256

        curr_item_ = self.imgs_[idx]

        obj_idx = curr_item_[:curr_item_.find(".")]
        path_mask_ = str(int(int(obj_idx) / 2000))
        mask_name_ = "0" * (5 - len(obj_idx)) + obj_idx + "_hair.png"

        curr_item_ = cv2.imread(os.path.join(self.path_data_, "CelebA-HQ-img", curr_item_))
        curr_item_ = cv2.resize(curr_item_, (256, 256))
        curr_item_ = cv2.cvtColor(curr_item_, cv2.COLOR_BGR2RGB) / 255.
        curr_mask_ = cv2.imread(os.path.join(self.masks_path_, path_mask_, mask_name_))
        curr_mask_ = cv2.resize(curr_mask_, (256, 256))
        curr_mask_ = np.sum(curr_mask_, -1, keepdims=True).astype(np.float32)

        curr_mask_ /= np.max(curr_mask_)
        curr_mask_ = curr_mask_.astype(np.int32)

        return np.expand_dims(curr_item_, 0), np.expand_dims(curr_mask_, 0)


class unet:
    def __init__(self, path_data, logdir="summary", epochs=5):
        self.initializer = "he_normal"
        self.model_ = None
        self.epochs_ = epochs

        # data preprocessing
        self.path_data_ = path_data

        self.imgs_all_ = os.listdir(os.path.join(path_data, "CelebA-HQ-img"))
        split = int(len(self.imgs_all_) * 0.9)
        np.random.shuffle(self.imgs_all_)
        self.train_data_ = tf.data.Dataset.from_generator(
                    segmGenerator(path_data, self.imgs_all_[:int(split/9)]),
                        output_types=(tf.float32, tf.int32),
                        output_shapes=((None,None,None,3), (None, None, None,1)))

        self.valid_data_ = tf.data.Dataset.from_generator(
                        segmGenerator(path_data, self.imgs_all_[split:]),
                        output_types=(tf.float32, tf.int32),
                        output_shapes=((None,None,None,3), (None, None, None,1)))

        self.steps_per_epoch_ = split
        self.initializer = "he_normal"
        self.logdir_ = logdir

    def encoder(self, x, scope):
        initializer = self.initializer
        with tf.name_scope("raw"):
            conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', 
                            kernel_initializer=initializer)(x)
            conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', 
                            kernel_initializer=initializer)(conv_enc_1)

            # Block encoder 2
            max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
            conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                    kernel_initializer = initializer)(max_pool_enc_2)
            conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                    kernel_initializer = initializer)(conv_enc_2)

            # Block  encoder 3
            max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
            conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                    kernel_initializer = initializer)(max_pool_enc_3)
            conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                    kernel_initializer = initializer)(conv_enc_3)

            # Block  encoder 4
            max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
            conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                        kernel_initializer = initializer)(max_pool_enc_4)
            conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                        kernel_initializer = initializer)(conv_enc_4)
            # -- Encoder -- #

            # ----------- #
            maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
            conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                        kernel_initializer = initializer)(maxpool)
            conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                        kernel_initializer = initializer)(conv)
            # ----------- #

        return conv_enc_1, conv_enc_2, conv_enc_3, conv_enc_4, conv

    def decoder(self, x, scope, N_CLASSES=1):
        initializer = self.initializer
        with tf.name_scope(scope):
            conv_enc_1, conv_enc_2, conv_enc_3, conv_enc_4, conv = x
            up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
            merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
            conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(merge_dec_1)
            conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(conv_dec_1)

            # Block decoder 2
            up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
            merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
            conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(merge_dec_2)
            conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(conv_dec_2)

            # Block decoder 3
            up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
            merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
            conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(merge_dec_3)
            conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(conv_dec_3)

            # Block decoder 4
            up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
            merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
            conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(merge_dec_4)
            conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(conv_dec_4)
            conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', 
                            kernel_initializer = initializer)(conv_dec_4)
            # -- Dencoder -- #

            output = Conv2D(N_CLASSES, 1, activation = 'sigmoid')(conv_dec_4)

        return output


    def build_raw_unet(self):
       x = Input(shape=[256, 256, 3])
       x_intermed = self.encoder(x, "raw")
       x_out = self.decoder(x_intermed, "raw", N_CLASSES=1)
       self.model_ = tf.keras.Model(inputs=x, outputs=x_out)
       self.model_.compile(optimizer=Adam(learning_rate=0.001), 
                loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy", self.iou])
       #print("=================================================")
       print(self.model_.summary()) 
       return self.model_


    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = intersection / (tf.reduce_sum(y_true_f)
             + tf.reduce_sum(y_pred_f) - intersection)
        return score

    def iou(self, y_true, y_pred):
        loss = self.dice_coeff(y_true, y_pred)
        return loss


    def build_enh_unet(self):
        spn = spnStage()

        x = Input(shape=[256, 256, 3])
        x_intermed = self.encoder(x, "raw")
        x_out = self.decoder(x_intermed, "raw", N_CLASSES=1)

        x_out_for_spn = self.decoder(x_intermed, "spn", N_CLASSES=4)
        x_out_model = spn.spnForward(x_out, x_out_for_spn)

        self.model_ = tf.keras.Model(inputs=x, outputs=x_out_model)
        self.model_.compile(optimizer=Adam(learning_rate=0.001), 
                loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy", self.iou])
        return self.model_

    def train(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir_, histogram_freq=1)
        callbacks_ = [tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            # to save checkpoints
            tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', 
                    verbose=1, save_best_only=True, save_weights_only=True)]

        history_ = self.model_.fit(self.train_data_, epochs=self.epochs_, 
                                steps_per_epoch=int(self.steps_per_epoch_/9), 
                                validation_steps=int(len(self.imgs_all_) * 0.1),
                                validation_data=self.valid_data_, callbacks=callbacks_)
        return history_


