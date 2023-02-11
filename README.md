# MarshBoundaries

Joint project with [Yiyang Xu](https://www.bu.edu/earth/profiles/xu/) to classify marshes into water, marsh and upland. Uses depth 1 convolutional net with kernel size 4, or equivalently classifies each pixel by applying linear regression to the 4x4 neighborhood of the image pixels. We didn't see gains by using deeper u-nets (the classic architecture for image segnemtation). Results:

![[linreg3000allsites.png]]
the figure is self-explanatory other than the dark line across the rightmost images. Above the line is test data (which the algorithm does not get to see during training time) and below it is training data. Note that lots of the pixels are unlabeled, so we obtain predictions for a lot of unlabeled area.

Here is the confusion matrix for the three classes. The i, j entry is the fraction of pixels in class i misclassified as class j. So for instance we see that 14% of water pixels are labeled as marsh by the algorithm. 

![[linreg3000confusion.png]]
We only trained the model on 3 images, one of each site. How does that make sense, you ask? Really each pixel is a data point, and we have 2048x2816 of those per image so it's actually a lot of data. Unsurprisingly, however, the model generalizes better within each site than to images of entirely new sites. 

# HOW TO USE

Clone this repository, install necessary dependencies for PredictFromSaved.py. Unfortunately you need to run the cells in MarshBoundariesTrain.ipynb up to and including saving the file numpymask. Then you can run

```
python PredictFromSaved.py
```

in your command line. Or, if you'd prefer, just fire up the jupyter notebook MarshBoundaryDemo.ipynb. 

Even though we augmented the data with random zooms, we recommend only using the classifier on images starting in landsat resolution and resized to 2048 x 2816. 
