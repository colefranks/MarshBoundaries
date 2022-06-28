import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import matplotlib.pyplot as plt

#convert image to tensor 

XDIM = 2048
YDIM = 11*XDIM//8

#for now 
FOLDER = "train_jpg/GCE"

#assumes the images in the folder look like Band1.jpg, Band2.jpg etc
def load_marsh_images(folder):
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
def get_tensor(images):
    images_tensor = tf.convert_to_tensor(images,dtype =tf.float32)
    SAMPLES,BANDS,HEIGHT,WIDTH = images_tensor.shape
    image_tensor = tf.transpose(images_tensor,[0,2,3,1])
    return image_tensor

#create mask from prediction

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

#load model 



#it is not robust to re-sizing.

def show_predictions(
    model_path = 'saved-models/Depth2Maxpool87val',
    folder_path = "train_jpg/GCE",
    startx = 0, 
    starty = 0, 
    zoomx = 400,
    zoomy=400,
    fig_size=5
    ):

    startx = min(XDIM,startx)
    starty = min(XDIM,startx)
    endx = min(XDIM,startx + zoomx)
    endy = min(YDIM,starty + zoomy)

    WHICH_IMG = 0
    WHICH_BAND = 4
    BANDS = 11

    model = tf.keras.models.load_model(model_path)

    images = load_marsh_images(folder_path)

    small_image = get_tensor(images[WHICH_IMG,:,startx:endx ,starty:endy].reshape([1,BANDS,zoomx,zoomy]))

    small_pred = model.predict(small_image)

    small_pred = create_mask(small_pred)


    plt.figure(figsize=(fig_size, fig_size))
    plt.subplot(121)
    plt.imshow(small_image[0,:,:,WHICH_BAND])
    plt.subplot(122)
    plt.imshow(small_pred,vmin=0, vmax=3)

    plt.show()

def main():
    show_predictions()

if __name__ == '__main__':
    main()
