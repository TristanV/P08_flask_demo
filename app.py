# Author : Tristan Vanrullen
# Licence : see Licence file at the root of the repository
# Project : Open Classrooms P08
# This repository is part of a project dedicated to image semantic segmentation


import numpy as np
import pandas as pd
  
import timeit  
from time import time

import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import ListedColormap 

import seaborn as sns 

import os.path
import os 
import pickle


import cv2
 
#tensorflow = 2.4 is important for compatibility with segmentation-models
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.backend as backend 
# from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
# from tensorflow.keras.utils import Sequence, to_categorical
# from tensorflow.keras.losses import binary_crossentropy

# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
# from tensorflow.keras.applications import VGG16 

# print("OpenCV version ",cv2.__version__)
# print("Tensorflow version ",tf.__version__)
# print("Keras version ",keras.__version__) 

from flask import Flask, render_template, request



# Flask app initialization
app = Flask(__name__)


@app.route('/')
def index():
    original_images = os.listdir('./static/original')
    return render_template('index.html',original_images=original_images)


@app.route('/prediction', methods=['POST']) 
def prediction():

    def overlay_mask_on_image(image, mask, cmap_colors):
        # in our case, cmap_colors is an array of normalized tuples (R,G,B,A) where R,G,B and A are floats between 0 and 1
        mask_image = np.zeros(mask.shape+(3,), dtype=np.uint8) 
        for i in range(len(cmap_colors)): 
            index_filter = (mask == i)
            mask_image[mask == i] = np.array(cmap_colors[i][:3])*255 # rebuild an RGB color 
        overlay = cv2.addWeighted(image, 0.3,mask_image, 0.8, 0)
        return overlay


    # 1 ---------------------------------------------------------------------
    # fetch the selected file in the POST request data
    file = request.form['file']
    original_image_path = str('./static/original/'+ file) 
    print("submitting inference job with image : "+original_image_path)
    original_img = cv2.imread(f'{original_image_path}')

    # 2 ---------------------------------------------------------------------
    # get the legend image and the colormap for our application
    legend_image_path =  str('./static/label_legend.png')
    
    df_categories = pd.read_pickle('./data/categories.pkl')
    labels=df_categories['category']
    colors=df_categories['color']
    
    # let's create a colormap for our categories
    # note that colors defined in the 'color' feature of the df_categories dataframe are defined as 4 coefficients between 0 and 1. [R,G,B,A]
    # each of these colors can be used as is to build a colormap, or have to be converted in a 255^4 space to be used as RGBA colors
    labelcateg_colormap_name = 'labelcateg_colormap'
    
    labelcateg_colormap = ListedColormap(df_categories['color'].tolist(), name=labelcateg_colormap_name)

    # debug #if ('labelcateg_colormap' not in plt.colormaps()): # avoid a warning
    # debug #    plt.register_cmap('labelcateg_colormap', labelcateg_colormap) # make the colormap available in the matplotlib 'colormaps' object
 
    
    # debug #cmap = cm.get_cmap(labelcateg_colormap_name, 8)
    cmap = labelcateg_colormap
    cmapcolors = cmap.colors
    
    # 3 ---------------------------------------------------------------------
    # convert ground truth image mask to our colormap
    label_image_path = str('./static/label/'+ file) 
    reflabel_img = cv2.imread(f'{label_image_path}', cv2.IMREAD_GRAYSCALE)     
    reflabel_img=overlay_mask_on_image(image=original_img,mask=reflabel_img,cmap_colors=cmapcolors)
    label_image_path = str('./static/label/reference.png') 
    plt.imsave(label_image_path, reflabel_img)
    
    # 4 ---------------------------------------------------------------------
    # generate prediction and colorize the result with our colormap
    prediction_image_path = str('./static/prediction/'+ file)
    model = tf.keras.models.load_model('./model/vgg_unet.h5') # , custom_objects={})
    
    # technique used in my notebook for the prediction and image reshape process on hand-coded models,
    # adapted to the VGG_UNET model that is not layered the same way because it comes from an open source module.
    # general parameters for model
    IMG_WIDTH            = 224
    IMG_HEIGHT           = 224
    # target_size for load_img step: either `None` (default to original size) or tuple of ints `(img_height, img_width)`
    TARGET_SIZE          = (IMG_HEIGHT,IMG_WIDTH) 
    NB_CLASSES           = len(df_categories)

    
    # print("model input shape:",model.input_shape)
    # >> model input shape: (None, 224, 224, 3) 
    input_width = model.input_shape[1]
    input_height =  model.input_shape[2] 
    
    # print("model ouput shape:",model.output_shape)
    # >> model ouput shape: (None, 12544, 8)
    # our model divides by 2 the size of the input image (224 = 112 * 2)
    # our model preserves the input width/height ratio, so output is 112 * 112 (=12544)
    output_width = 112 
    output_height = 112 
    colors = [[i,i,i] for i in np.arange(0,NB_CLASSES)] # we will colorize the masks by ourselves, so we just want output masks saved as grayscale images

    #colors=df_categories['color']
    
    original_img = load_img(f'{original_image_path}')
    original_size=original_img.size 
    # print("original size:",original_size)
    # >> original size: (2048, 1024)
    
    input_image = original_img.resize(TARGET_SIZE)
    # print("resized size :",input_image.size)
    # >> resized size : (224, 224)
    
    input_image =  img_to_array(input_image)
    # the following normalization was used in some of our models (the hand coded ones), but not for the vgg_unet model
    # input_image =  img_to_array(input_image)/255 # to normalize image to floats between 0.0 and 1.0
    
    pr = model.predict(np.array([input_image]))[0] # we process a 1 image array, so we output a 1 mask array 
    # print("unique predictied labels :" , np.unique(pr,return_counts=True)) 
    # >> unique predictied labels : (array([2.4147025e-16, 2.4150249e-16, 5.4166391e-16, ..., 9.9999976e-01,
    # >>    9.9999988e-01, 1.0000000e+00], dtype=float32), array([  1,   1,   1, ...,  47,  65, 151], dtype=int64))
    # print(pr.size)
    # >> 100352
    # print(pr.shape)
    # >> (12544, 8)
    
    pr = pr.reshape((output_height,  output_width, NB_CLASSES)).argmax(axis=2)
     
    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(NB_CLASSES):
        seg_arr_c = pr[:, :] == c  
        # print("class=",c,seg_arr_c)
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
 
    seg_img = cv2.resize(seg_img, original_size, interpolation=cv2.INTER_NEAREST) 
    
    cv2.imwrite(prediction_image_path, seg_img) 
    
    # 5 ---------------------------------------------------------------------
    # finally colorize the resulting prediction mask and save it too
    original_img = cv2.imread(f'{original_image_path}')
    prediction_img = cv2.imread(f'{prediction_image_path}', cv2.IMREAD_GRAYSCALE)     
    # prediction_img = cv2.resize(prediction_img, original_size, interpolation=cv2.INTER_NEAREST) 
    prediction_img=overlay_mask_on_image(image=original_img,mask=prediction_img,cmap_colors=cmapcolors)
    prediction_image_path = str('./static/prediction/prediction.png') 
    plt.imsave(prediction_image_path, prediction_img)
     
    
    return render_template('prediction.html', 
        original_image_path =original_image_path,
        label_image_path=label_image_path,
        prediction_image_path=prediction_image_path,
        legend_image_path=legend_image_path)

    
# Run app
if __name__ == '__main__':
    app.run(debug=True)
    
    



