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
class segmGeneratorSimple:
    def __init__(self, path_data, images, batch_size=1):
        self.path_data_ = path_data
        self.batch_size_ = batch_size
        self.cut = 32

        self.imgs_ = images#os.listdir(os.path.join(path_data, "CelebA-HQ-img"))
        self.masks_ = os.path.join(path_data, "images-gt")
        self.n = len(self.imgs_)
        self.i = -1

    def __len__(self):
        return len(self.imgs_)

    def __next__(self):
        self.i += 1
        if self.i == self.n:
            raise StopIteration()
        else:
            return self.__getitem__(self.i)

    def __iter__(self):
        return self

    def __call__(self):
        self.i = 0
        return self

    def __getitem__(self, idx):
        curr_item_ = self.imgs_[idx]

        obj_idx = curr_item_[:curr_item_.find(".")]
        #path_mask_ = str(int(int(obj_idx) / 2000))
        mask_name_ = curr_item_[: \
                        curr_item_.rfind(".")] + ".png"
        #"0" * (5 - len(obj_idx)) + obj_idx + "_hair.png"

        curr_item_ = cv2.imread(os.path.join(self.path_data_, "images", curr_item_))
        #curr_item_ = cv2.resize(curr_item_, (128, 128))
        curr_item_ = cv2.cvtColor(curr_item_, cv2.COLOR_BGR2RGB) / 255.
        dx = self.cut - curr_item_.shape[0] % self.cut
        dx = 0 if dx == self.cut else dx

        dy = self.cut - curr_item_.shape[1] % self.cut
        dy = 0 if dy == self.cut else dy
        #print("----------", dx, dy, curr_item_.shape)
        if dx > 0:
            curr_item_ = np.concatenate([curr_item_, 
				np.zeros((dx, curr_item_.shape[1], 3))], axis=0)
        if dy > 0:
            curr_item_ = np.concatenate([curr_item_, 
				np.zeros((curr_item_.shape[0], dy, 3))], axis=1)

        curr_mask_ = cv2.imread(os.path.join(self.masks_, mask_name_))
        curr_mask_ = np.sum(curr_mask_, -1, keepdims=False).astype(np.float32)
        curr_mask_ /= np.max(curr_mask_)
        curr_mask_ = curr_mask_.astype(np.int32)
        if dx > 0:
            curr_mask_ = np.concatenate([curr_mask_, 
				np.zeros((dx, curr_mask_.shape[1])).astype(np.int32)], axis=0)
        if dy > 0:
            curr_mask_ = np.concatenate([curr_mask_, 
             np.zeros((curr_mask_.shape[0], dy)).astype(np.int32)], axis=1)
        #print(curr_item_.shape, curr_mask_.shape, "=================================")
        #print(np.mean(curr_mask_))

        return np.expand_dims(curr_item_, 0), np.expand_dims(curr_mask_, 0)


class unet:
    def __init__(self, path_data, logdir="summary", epochs=5, n_classes=2, model_name="unet.h5"):
        self.initializer = "he_normal"
        self.model_ = None
        self.epochs_ = epochs
        self.N_CLASSES = n_classes
        self.model_name_ = model_name

        # data preprocessing
        self.path_data_ = path_data

        self.imgs_all_ = os.listdir(os.path.join(path_data, "images"))
        split = int(len(self.imgs_all_) * 0.9)
        np.random.shuffle(self.imgs_all_)
        self.train_data_ = tf.data.Dataset.from_generator(
                    segmGeneratorSimple(path_data, self.imgs_all_[:int(split)]),
                        output_types=(tf.float32, tf.int32),
                        output_shapes=((None,None,None,3), (None, None, None)))

        self.valid_data_ = tf.data.Dataset.from_generator(
                        segmGeneratorSimple(path_data, self.imgs_all_[split:]),
                        output_types=(tf.float32, tf.int32),
                        output_shapes=((None,None,None,3), (None, None, None)))

        self.steps_per_epoch_ = split
        self.initializer = "he_normal"
        self.logdir_ = logdir
        self.optimizer = tf.optimizers.Adam( learning_rate=0.001 )

        # checkpoint manager
        self.cpkt_ = None
        self.manager_ = None


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

    def decoder(self, x, scope, nclassout):
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
            #conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', 
            #                kernel_initializer = initializer)(conv_dec_4)
            # -- Dencoder -- #

            output = Conv2D(nclassout, 1, activation = 'softmax')(conv_dec_4)
            output = tf.clip_by_value(output, 0.00001, 0.99999)
            #print(output)
        return output

    def build_nn(self, type_arch="unet_weak"):
        if type_arch == "unet_weak":
            self.build_raw_unet()
        elif type_arch == "unet_enh":
            self.build_enh_unet()
        #self.cpkt_ = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer,
        #        net=self.model_, iterator=iter(self.train_data_))
        #self.manager_ = tf.train.CheckpointManager(self.cpkt_, "./model", max_to_keep=3)

    def build_raw_unet(self):
       x = Input(shape=[None, None, 3])
       x_intermed = self.encoder(x, "raw")
       x_out = self.decoder(x_intermed, "raw", self.N_CLASSES)
       self.model_ = tf.keras.Model(inputs=x, outputs=x_out)
       self.model_.compile()#optimizer=Adam(learning_rate=0.001), 
       #         loss=tf.keras.losses.SparseCategoricalCrossentropy(
       #           reduction=tf.keras.losses.Reduction.SUM), metrics=["accuracy", self.iou])
       #print("=================================================")
       print(self.model_.summary()) 
       return self.model_


    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.cast(tf.reduce_sum(y_true_f * y_pred_f), dtype=tf.float32)
        score = intersection / (tf.cast(tf.reduce_sum(y_true_f), dtype=tf.float32)
             + tf.cast(tf.reduce_sum(y_pred_f), dtype=tf.float32) - intersection)
        return score

    def iou(self, y_true, y_pred):
        loss = self.dice_coeff(y_true, y_pred)
        return loss


    def build_enh_unet(self):
        spn = spnStage()

        x = Input(shape=[None, None, 3])
        x_intermed = self.encoder(x, "raw")
        x_out = self.decoder(x_intermed, "raw", self.N_CLASSES)

        x_out_for_spn = self.decoder(x_intermed, "spn", self.N_CLASSES*4)
        x_out_model = spn.spnForward(x_out, x_out_for_spn)

        self.model_ = tf.keras.Model(inputs=x, outputs=x_out_model)
        self.model_.compile()#optimizer=Adam(learning_rate=0.001), 
        #        loss=tf.keras.losses.SparseCategoricalCrossentropy(
        #         reduction=tf.keras.losses.Reduction.SUM), metrics=["accuracy", self.iou])
        return self.model_

    def loss(self, pred, target):
        #tf.print(pred.shape, target.shape, "------------------------------")
        local_loss = tf.keras.losses.sparse_categorical_crossentropy(target, pred)
        return tf.reduce_mean(local_loss)

    def train_step(self, inputs, outputs):
        current_loss = 0
        with tf.GradientTape() as tape:
            #tf.print(inputs.shape, "--------------------")
            current_loss = self.loss(self.model_(inputs), outputs)
        tf.print(current_loss)
        grads = tape.gradient(current_loss, self.model_.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_.trainable_variables))

    def train(self):
        for epo_ in range(self.epochs_):
            self.train_epoch()
            tf.keras.models.save_model(self.model_, "model")
            #self.model_.save_weights("model/my_cp")
            #self.manager_.write()
            #self.manager_.update_state()
            #self.manager_.save()

    @tf.function
    def train_epoch(self):
        for features in self.train_data_:
            image, label = features
            self.train_step(image, label)
            #self.cpkt_.step.assign_add(1)
            #self.manager_.save()

        for features in self.valid_data_:
            image, label = features
            net_out = self.model_(image)
            curr_loss = self.loss(net_out, label)
            tf.print("valid", curr_loss, self.iou(label, 
                          tf.cast(tf.math.argmax(net_out, axis=-1), dtype=tf.int32)))

    """
    def train(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir_, histogram_freq=1)
        callbacks_ = [tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            # to save checkpoints
            tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', 
                    verbose=1, save_best_only=True, save_weights_only=True)]

        history_ = self.model_.fit(self.train_data_, epochs=self.epochs_, 
                                steps_per_epoch=int(self.steps_per_epoch_), 
                                validation_steps=int(len(self.imgs_all_) * 0.1),
                                validation_data=self.valid_data_, callbacks=callbacks_)
        return history_
    """

