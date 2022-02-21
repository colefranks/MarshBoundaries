# MarshBoundaries
Using aerial marsh photos to determine boundaries between land and water

Work in progress. Goal is to produce this:

<img src="plum_boundary.jpg"/>

from this:

<img src="plum_raw.jpg"/>

So far we are just using small patches as data points and inferring as follows:

<img src="1000-epochs.png"/>

You can see that it looks pretty bad. Next steps are to use a mask which labels marsh/nonmarsh etc, not just the boundary. Also we can augment and obtain more data. Finally we can try a model with fewer trainable parameters, as the test images look good enough to indicate overfitting.

