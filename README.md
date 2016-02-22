<<<<<<< HEAD
# Neural Network
Author: Jakub Cisło
http://cislo.net.pl
jakub@cislo.net.pl

Implementation of neural network in Python using Numpy and Autograd.
Method: transformation matrix and stochastic gradient descent
**Required**: `python2`, `numpy`, `matplotlib`, `autograd`
	and MNIST and CIFAR-10 datasets

##Usage:
`python -i mnist.py`
`python -i cifar.py`

###Interactive commands
* `create(layers=[28*28, 100, 10], batch_size=32, dropout=0.1)` - create new model
* `learn()` - let model learn
* `save(name)` - save model to file
* `load(name)` - load model from file
* `info()` - show statistics
* `best()` - show best images from each category
* `worst()` - show worst images from each category

