import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import matplotlib.pyplot as plt

#convert image to tensor 

XDIM = 2048
YDIM = 11*XDIM//8

#XDIM = 512
#YDIM = 11*XDIM//8


#for now 
#FOLDER = "train_jpg/VCR"

#assumes the images in the folder look like Band1.jpg, Band2.jpg etc
def folder_to_array(folder):
    images = []
    
    for i in range(1,12):
        img = cv.imread(os.path.join(folder,"Band%d.jpg" % i))
        #print("Band%d.jpg" % i)
        if img is not None:
            #convert to black and white
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img,(YDIM, XDIM))
            images.append(img)
    images = np.array([images])
    return images

#convert image to tensor that can go into model
def get_tensor_pred(images):
    images_tensor = tf.convert_to_tensor(images,dtype =tf.float32)
    SAMPLES,BANDS,HEIGHT,WIDTH = images_tensor.shape
    image_tensor = tf.transpose(images_tensor,[0,2,3,1])
    return image_tensor

#create mask from prediction

def create_mask_pred(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

#load model 

CLASS_DICT = {0: 'marsh', 1: 'water', 2: 'upland', 3: 'unlabeled'}

#it is not robust to re-sizing.

def pred_from_saved(
    model_path = 'saved-models/Depth2RandomFlip89val',
    folder_path = "train_jpg/PIE",
    startx = 0, 
    starty = 0, 
    zoomx = 1000,
    zoomy=1000,
    fig_size=5
    ):

    startx = min(XDIM,startx)
    starty = min(XDIM,startx)
    endx = min(XDIM,startx + zoomx)
    endy = min(YDIM,starty + zoomy)

    zoomx = min(XDIM,zoomx)
    zoomy = min(YDIM,zoomy)

    WHICH_IMG = 0
    WHICH_BAND = 4
    BANDS = 11

    model = tf.keras.models.load_model(model_path)

    images = folder_to_array(folder_path)

    small_image = get_tensor_pred(images[WHICH_IMG,:,startx:endx ,starty:endy].reshape([1,BANDS,zoomx,zoomy]))

    small_1hot = model.predict(small_image)

    small_pred = create_mask_pred(small_1hot)

    num_class = len(CLASS_DICT)


    fig = plt.figure(figsize=(fig_size, fig_size))
    ax1 = fig.add_subplot(121)
    ax1.imshow(small_image[0,:,:,WHICH_BAND])
    cmap = plt.cm.get_cmap('viridis', num_class)
    ax2= fig.add_subplot(122)
    im2 = ax2.imshow(small_pred,vmin=0, vmax=num_class-1,cmap=cmap)
    cbar = plt.colorbar(
        im2, 
        ticks=[3/8 + i*((num_class-1.0)/num_class) for i in range(num_class)],
        orientation='horizontal',ax=fig.get_axes())
    cbar.set_ticklabels([CLASS_DICT[i] for i in range(num_class)])
    plt.show()
    return (small_image, small_1hot)

def main():
    pred_from_saved()

if __name__ == '__main__':
    main()
