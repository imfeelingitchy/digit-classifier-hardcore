# Purist Digit Classifier
## A feed-forward neural network coded using only Python and Numpy. No ML libraries.

Everything here is thanks to [this book](http://neuralnetworksanddeeplearning.com/index.html). I've commented almost every line in `network.py` to explain everything that's going on to the best of my abilities. I've also edited some lines for efficiency. Further, the MNIST loader is altered to use the data from the [original website](https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/).

By using just 30 hidden layers and 5 epochs, we get an accuracy of 94%. By increasing these numbers or tuning the learning rate, we could probably bring this up to 97% with the current architecture.

![](https://github.com/imfeelingitchy/digit-classifier-hardcore/blob/main/images/screenshot.png)
