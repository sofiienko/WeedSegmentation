import skimage.transform as trans
import numpy as np
import keras
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import *
from keras.layers import *

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,EarlyStopping
from keras import backend as keras

def unet_sigmoid(pretrained_weights = None,input_size = (256,256,3)):
    from keras.models import Model, load_model, save_model
    from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras import backend as K
    from keras import optimizers
    import tensorflow as tf
    from keras.preprocessing.image import array_to_img, img_to_array, load_img

    def BatchActivate(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation==True: x = BatchActivate(x)
        return x

    def residual_block(blockInput, num_filters=16, batch_activate=False):
        x = BatchActivate(blockInput)
        x = convolution_block(x, num_filters, (3,3))
        x = convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        if batch_activate: x = BatchActivate(x)
        return x

        # Build Model
    def build_model(input_layer, start_neurons, DropoutRatio=0.5):
        # 101 -> 50
        conv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(input_layer)
        conv1 = residual_block(conv1, start_neurons*1)
        conv1 = residual_block(conv1, start_neurons*1, True)
        pool1 = MaxPooling2D((2,2))(conv1)
        pool1 = Dropout(DropoutRatio/2)(pool1)
        
        # 50 -> 25
        conv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(pool1)
        conv2 = residual_block(conv2, start_neurons*2)
        conv2 = residual_block(conv2, start_neurons*2, True)
        pool2 = MaxPooling2D((2,2))(conv2)
        pool2 = Dropout(DropoutRatio)(pool2)
        
        # 25 -> 12
        conv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(pool2)
        conv3 = residual_block(conv3, start_neurons*4)
        conv3 = residual_block(conv3, start_neurons*4, True)
        pool3 = MaxPooling2D((2,2))(conv3)
        pool3 = Dropout(DropoutRatio)(pool3)
        
        # 12 -> 6
        conv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(pool3)
        conv4 = residual_block(conv4, start_neurons*8)
        conv4 = residual_block(conv4, start_neurons*8, True)
        pool4 = MaxPooling2D((2,2))(conv4)
        pool4 = Dropout(DropoutRatio)(pool4)
        
        # Middle
        convm = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
        convm = residual_block(convm, start_neurons*16)
        convm = residual_block(convm, start_neurons*16, True)
        
        # 6 -> 12
        deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(DropoutRatio)(uconv4)
        
        uconv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(uconv4)
        uconv4 = residual_block(uconv4, start_neurons*8)
        uconv4 = residual_block(uconv4, start_neurons*8, True)
        
        # 12 -> 25
        deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding='same')(uconv4)
        print(f"deconv3 {deconv3.shape} - {start_neurons*4}")
        print(f"conv3 {conv3.shape}")
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(DropoutRatio)(uconv3)
        
        uconv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(uconv3)
        uconv3 = residual_block(uconv3, start_neurons*4)
        uconv3 = residual_block(uconv3, start_neurons*4, True)
        
        # 25 -> 50
        deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding='same')(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(DropoutRatio)(uconv2)
        
        uconv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(uconv2)
        uconv2 = residual_block(uconv2, start_neurons*2)
        uconv2 = residual_block(uconv2, start_neurons*2, True)
        
        # 50 -> 101
        deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding='same')(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(DropoutRatio)(uconv1)
        
        uconv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(uconv1)
        uconv1 = residual_block(uconv1, start_neurons*1)
        uconv1 = residual_block(uconv1, start_neurons*1, True)
        
        output_layer_noActi = Conv2D(1, (1,1), padding='same', activation=None)(uconv1)
        output_layer = Activation('sigmoid')(output_layer_noActi)
        
        return output_layer


    input_layer = Input(input_size)
    output_layer = build_model(input_layer, 16,0.5)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = 0.005)
    #model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
    model1.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model1.summary()

    if(pretrained_weights):
        print(f"MODEL LOADED {pretrained_weights}")
        model1.load_weights(pretrained_weights)

    return model1







###############################################OLD#############################
def unet_sigmoid_Old(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        print(f"MODEL LOADED {pretrained_weights}")
        model.load_weights(pretrained_weights)
    return model
	
	
	
	
	
	