import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def dice_coef(y_true, y_pred):
  y_true_f = keras.flatten(y_true)
  y_pred_f = keras.flatten(y_pred)
  intersection = keras.sum(y_true_f * y_pred_f)
  return (2. * intersection + 0.0001) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 0.0001)

def dice_coef_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred)

def unet(pretrained_weights=None, input_size=(512, 512, 1), use_jaccard = False, learning_rate=0.0001):
    inputs = Input(input_size)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2),interpolation='bilinear')(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)

    up7 = Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2),interpolation='bilinear')(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)

    up8 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2),interpolation='bilinear')(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)

    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2),interpolation='bilinear')(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss= [dice_coef_loss if use_jaccard else  "binary_crossentropy"],
        metrics=['accuracy',dice_coef],
    )

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
