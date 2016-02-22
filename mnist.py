#!/usr/bin/env python2
# Author: Jakub Cislo
# http://cislo.net.pl
# jakub@cislo.net.pl
# License: MIT
# Copyright (C) 2016

import utils
from neuralnetwork import NeuralNetworkModel
import autograd.numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import plt_image

TRAINING_IMAGES_PATH = 'mnistset/train-images-idx3-ubyte.gz'
TRAINING_LABELS_PATH = 'mnistset/train-labels-idx1-ubyte.gz'
TESTING_IMAGES_PATH = 'mnistset/t10k-images-idx3-ubyte.gz'
TESTING_LABELS_PATH = 'mnistset/t10k-labels-idx1-ubyte.gz'
MODELS_PATH = 'mnistmodels/'

print 'Reading images from %s' % TRAINING_IMAGES_PATH
images = utils.read_images(TRAINING_IMAGES_PATH)
print 'Reading labels from %s' % TRAINING_LABELS_PATH
labels = utils.read_labels(TRAINING_LABELS_PATH)
print 'Reading images from %s' % TESTING_IMAGES_PATH
testing_images = utils.read_images(TESTING_IMAGES_PATH)
print 'Reading labels from %s' % TESTING_LABELS_PATH
testing_labels = utils.read_labels(TESTING_LABELS_PATH)
training_images = images[5000:-5000]
training_labels = labels[5000:-5000]
validating_images = np.concatenate([images[:5000], images[-5000:]])
validating_labels = np.concatenate([labels[:5000], labels[-5000:]])

np.set_printoptions(suppress=True)
nnm = None


def create(layers=[28*28, 100, 10], batch_size=32, dropout=0.1):
    print 'Creating neural network'
    global nnm
    nnm = NeuralNetworkModel(layers, batch_size, dropout)


def learn():
    print 'Learning'
    for no, _ in enumerate(nnm.epochs_learn(training_images, training_labels)):
        print 'Epoch {0}: {1}'.format(no, nnm.test(validating_images, validating_labels))


def save(name):
    print 'Saving neural network'
    with open(MODELS_PATH + name + '.pickle', 'wb') as f:
        pickle.dump(nnm, f)


def load(name):
    print 'Loading neural network'
    global nnm
    with open(MODELS_PATH + name + '.pickle', 'rb') as f:
        nnm = pickle.load(f)


def info():
    print nnm
    print 'Final result: {0}'.format(nnm.test(testing_images, testing_labels))


def best():
    ans = nnm.answer(testing_images)
    correct = (np.argmax(ans, 1) == np.argmax(testing_labels, 1))
    for digit in range(10):
        idx = np.argmax(ans[correct][:, digit])
        plt.subplot(2, 5, digit+1)
        plt_image(testing_images[correct][idx].reshape([28, 28]))
        print digit, ans[correct][idx][digit]
    plt.show()


def worst():
    ans = nnm.answer(testing_images)
    for digit in range(10):
        where = np.argmax(testing_labels, 1) == digit
        idx = np.argmin(ans[where, digit])
        plt.subplot(2, 5, digit+1)
        plt_image(testing_images[where][idx].reshape([28, 28]))
        print digit, ans[where][idx]
    plt.show()

