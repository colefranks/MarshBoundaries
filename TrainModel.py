import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output


#LOAD IMAGES
# chose this size so that it approximates images well but is also highly divisible by 2

XDIM = 2048
YDIM = 11*XDIM//8 

def load_marsh_images(folder):
    images = []
    
    for i in range(1,12):
        img = cv.imread(os.path.join(folder,"Band%d.jpg" % i))
        #print("Band%d.jpg" % i)
        if img is not None:
            #convert to black and white
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img,(YDIM, XDIM))
            #warp so the mask is square
            #img = cv.warpPerspective(img,Mpad,(SIZE+2*PAD,SIZE+2*PAD))
            images.append(img)
    return images

#get images with PIE thrown out

gce_imgs = load_marsh_images("train_jpg/GCE")
#pie_imgs = load_marsh_images("train_jpg/PIE")
vcr_imgs = load_marsh_images("train_jpg/VCR")
images = np.array([
    gce_imgs,
    #pie_imgs
    vcr_imgs])
NUM_IMG = images.shape[0]




#GET MASKS (not PIE)
gce_mask = cv.imread("new-masks/updated_GCE_mask.jpg")
#pie_mask = cv.imread("new-masks/updated_PIE_mask.jpg")
vcr_mask = cv.imread("new-masks/updated_VCR_map.jpg")

new_masks = [cv.cvtColor(gce_mask, cv.COLOR_BGR2GRAY), 
         #cv.cvtColor(pie_mask, cv.COLOR_BGR2GRAY), 
         cv.cvtColor(vcr_mask, cv.COLOR_BGR2GRAY)]
masks = []
for m in new_masks:
    m = cv.resize(m,(YDIM,XDIM))
    masks.append(m)
new_masks = masks


#PREPROCESS MASKS
#change the values in the mask to integers from 1 to 5 instead of the weird 0-255 values currently in there
THRESH = [90,120,140,180] #found by exploring the data
new_masks = np.array(new_masks)
tidal_flat = (new_masks <= THRESH[0])
marsh = np.logical_and(THRESH[0] < new_masks, new_masks <= THRESH[1])
channel = np.logical_and(THRESH[1] < new_masks, new_masks <= THRESH[2])
upland = np.logical_and(THRESH[2] < new_masks, new_masks <= THRESH[3])
unlabeled = THRESH[3] < new_masks

#to throw away tidal_flat
unlabeled = np.logical_or(tidal_flat,unlabeled)
masks = [marsh,channel,upland,unlabeled]

int_mask = np.zeros_like(new_masks,dtype=int)
for i,m in enumerate(masks):
    int_mask = int_mask + i*m.astype(int)
masks = np.reshape(int_mask, (NUM_IMG,1,XDIM,YDIM))

#get frequencies of each class
ind,counts = np.unique(masks,return_counts=True)



#TEST AND TRAIN SPLITS
#I do a naive thing where I just convert the upper 5/8 of the mask to the unknown class.

not_care = np.max(masks)

te_masks = np.zeros_like(masks)
tr_masks = np.zeros_like(masks)

te_masks.fill(not_care)
tr_masks.fill(not_care)

te_masks[:,0,:3*XDIM//8,:]=masks[:,0,:3*XDIM//8,:]
tr_masks[:,0,3*XDIM//8:,:]=masks[:,0,3*XDIM//8:,:]


def get_tensor(images):
    images_tensor = tf.convert_to_tensor(images,dtype =tf.float32)
    SAMPLES,BANDS,HEIGHT,WIDTH = images_tensor.shape
    image_tensor = tf.transpose(images_tensor,[0,2,3,1])
    return image_tensor

train_images = get_tensor(images)
train_masks = get_tensor(tr_masks)

test_images = get_tensor(images)

test_masks = get_tensor(te_masks)



#CREATE TENSORFLOW DATASET (specialized format to speed up training)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_masks))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_masks))

#augmentations, for when we want to try that. not sure if this code works yet.

#data_augmentation = tf.keras.Sequential([
#  layers.RandomFlip("horizontal_and_vertical"),
#  layers.RandomRotation(0.2),
#])

#apply the augmentations; no idea why this num_parallel_calls is there
#train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

train_batches = (
    train_dataset
    .cache()
    .shuffle(NUM_IMG)
    .batch(NUM_IMG)
    #.repeat()
    #.map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_dataset.batch(3)




# DEFINE MODEL

#best one so far

def shallow_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[None, None, 11])
    
    #down
    x = tf.keras.layers.Conv2D(filters=5,kernel_size=2,strides=1,activation="relu",padding='same')(inputs)
    
    #maxpool
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
    
    x = tf.keras.layers.Conv2D(10,2,1,activation="relu",padding='same')(x)
    
    #maxpool
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
    
    #up
    x = tf.keras.layers.Conv2DTranspose(5,2,2,activation="relu")(x)
    
    last = tf.keras.layers.Conv2DTranspose(
    filters=5, kernel_size=2, strides=2,
    padding='same')
    
    x = last(x)
    
    #maxpool
    x = tf.keras.layers.Conv2D(output_channels,2,1,activation="relu",padding='same')(x)
    #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)

    
    return tf.keras.Model(inputs=inputs, outputs=x)


#COMPILE THE MODEL

#4 classes when tidal flats are unlabeled 

OUTPUT_CLASSES = 4
#OUTPUT_CLASSES=5

model = shallow_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              weighted_metrics=['accuracy']
             )



#VIEWING MODEL OUTPUT

for image,mask in test_dataset.take(1):
    sample_image,sample_mask = image, mask
for image,mask in train_dataset.take(1):
    sample_train_image,sample_train_mask = image, mask

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image, mask[0,0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def show_train_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0,0], create_mask(pred_mask)])
  else:
    display([sample_train_image, sample_train_mask,
             create_mask(model.predict(sample_train_image[tf.newaxis, ...]))])
    
def display(display_list):
  plt.figure(figsize=(20, 20))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1,len(display_list),i+1)
    plt.title(title[i])
    if display_list[i].shape[-1]==1:
        mi = 0
        ma = 3
    else:
        mi = 4
        ma = 7
    plt.imshow(tf.keras.utils.array_to_img(display_list[i][:,:,mi:ma]))
    plt.axis('off')
  plt.show()




#TRAINING MODEL

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 5==0:
        clear_output(wait=True)
        show_predictions()
    #show_train_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
tb_callback = tf.keras.callbacks.TensorBoard('./logs/Demo', update_freq=5)

## adding sample weights so that the final class is ignored 

def add_sample_weights(image, label):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tf.constant([1/counts[0],1/counts[1],1/counts[2], 0.0])
  #class_weights = tf.constant(inv_freq)
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights

#actually training 

EPOCHS = 10

#change to 3000 for real deal

model_history = model.fit(train_batches.map(add_sample_weights), 
                          #class_weight = {0:1,1:1,2:1,3:1,4:0},
                          epochs=EPOCHS,
                          #steps_per_epoch=STEPS_PER_EPOCH,
                          #validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches.map(add_sample_weights),
                          callbacks=[DisplayCallback(),tb_callback]
                         )


#the displayed training images have to be manually exited or something? gotta fix