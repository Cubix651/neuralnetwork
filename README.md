# Neural Network
Author: Jakub Cisło <br>
http://cislo.net.pl <br>
jakub@cislo.net.pl

Implementation of neural network in Python using Numpy and Autograd. <br>
Method: transformation matrix and stochastic gradient descent <br>
**Required**: `python2`, `numpy`, `matplotlib`, `autograd`, MNIST and CIFAR-10 datasets

##Usage:
`python -i mnist.py` <br>
`python -i cifar.py`

###Interactive commands
* `create(layers=[28*28, 100, 10], batch_size=32, dropout=0.1)` - create a new model
* `learn()` - start the model learning
* `save(name)` - save the model to file
* `load(name)` - load the model from file
* `info()` - show info and statistics about the model
* `best()` - show best images from each category
* `worst()` - show worst images from each category

