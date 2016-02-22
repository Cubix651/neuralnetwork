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
from utils import plt_image_color

TRAINING_DATA_PATHS = ['cifarset/data_batch_{0}'.format(no) for no in range(1, 6)]
TESTING_DATA_PATH = 'cifarset/test_batch'
TESTING_LABELS_PATH = 'mnistset/t10k-labels-idx1-ubyte.gz'
MODELS_PATH = 'cifarmodels/'

images = []
labels = []

for path in TRAINING_DATA_PATHS:
    print 'Reading data from %s' % path
    i, l = utils.read_cifar_set(path)
    images.append(i)
    labels.append(l)

images = np.concatenate(images)
labels = np.concatenate(labels)

print 'Reading data from %s' % TESTING_DATA_PATH
testing_images, testing_labels = utils.read_cifar_set(TESTING_DATA_PATH)
training_images = images[5000:-5000]
training_labels = labels[5000:-5000]
validating_images = np.concatenate([images[:5000], images[-5000:]])
validating_labels = np.concatenate([labels[:5000], labels[-5000:]])

np.set_printoptions(suppress=True)
nnm = None


def create(layers=[3072, 100, 10], batch_size=32, dropout=0.1):
    print 'Creating neural network'
    global nnm
    nnm = NeuralNetworkModel(layers, batch_size, dropout)


def learn():
    print 'Learning'
    for no, _ in enumerate(nnm.epochs_learn(training_images, training_labels, lambda x: x)):
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
        plt_image_color(testing_images[correct][idx].reshape([3, 32, 32]).transpose(1,2,0))
        print digit, ans[correct][idx][digit]
    plt.show()


def worst():
    ans = nnm.answer(testing_images)
    for digit in range(10):
        where = np.argmax(testing_labels, 1) == digit
        idx = np.argmin(ans[where, digit])
        plt.subplot(2, 5, digit+1)
        plt_image_color(testing_images[where][idx].reshape([3, 32, 32]).transpose(1,2,0))
        print digit, ans[where][idx]
    plt.show()

