import numpy as np
import skimage.transform as trans
import keras
from keras.preprocessing.image import img_to_array
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Unet import unet_sigmoid

def plot_image_from_array(img_array):
    imgplot = plt.imshow(img_array)
    plt.axis('off')
    plt.show()

def load_model():
    model = unet_sigmoid('TrainedModels/kernel_sigmoid_2702_batch16_weed-1556797625.7286613-weights.h5')
    return model
    #model = unet_sigmoid('sigmoid_2702_weed-1551286482.9070258-weights.h5')
	  
    

def array_convert_to_binary_mask(array ,threshold, shape = (256,256,3)):
  mask = np.zeros(shape)
  for i in range(len(array)):
    for j in range(len(array[i])):
      if array[i,j][0] > threshold:
        mask[i,j][0] = 1
      
  return mask


def combine_image_and_binary_mask(mask,threshold,img):
    print(f'inpyut: {img.shape}')
    
    img = np.copy(img)
    for i in range(len(mask)):
      for j in range(len(mask[i])):
        if mask[i,j][0] > threshold:
          img[i,j][0]+=0.5
      
    return img



def resize_img(img, new_shape = (256,256,3)):
    print(f"Image type:{type(img)}")
    return trans.resize(img, new_shape)


def prepare_image_for_model(raw_s,new_shape = (256,256,3)):
    z = np.zeros(new_shape)

    X_batch = []
    X_batch.append(z)
    X_batch.append(raw_s)
    X=np.asarray(X_batch, dtype=np.float32)
    #plot_image_from_array(X[1,:,:,:])
    return X

def prediction_to_image(prediction,image, threshold ):
    imageArray =  combine_image_and_binary_mask(prediction[1,:,:,:],threshold,image)
    #plot_image_from_array(imageArray)
    return imageArray


def prediction_to_mask(prediction,threshold, shape = (256,256,3)):
    imageArray = array_convert_to_binary_mask(prediction[1,:,:,:],threshold,shape = (256,256,3))
    #plot_image_from_array(imageArray)
    return imageArray


