# MarshBoundaries
Using aerial marsh photos to determine boundaries between land and water

Work in progress. Goal is to produce this:

<img src="plum_boundary.jpg"/>

from this:

<img src="plum_raw.jpg"/>

So far we are just using small patches as data points and inferring as follows:

<img src="1000-epochs.png"/>

You can see that there are many artifacts and that the boundary does not always look like a boundary. The next step is to enforce manually that the output is a boundary and use a custom loss function to compare boundaries.

