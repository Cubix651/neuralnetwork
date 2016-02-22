#!/usr/bin/env python2
# Author: Jakub Cislo
# http://cislo.net.pl
# jakub@cislo.net.pl
# License: MIT
# Copyright (C) 2016

import autograd.numpy as np
from autograd import grad
from utils import normalize, relu, dropout, softmax, logsoftmax, scale_and_rotate_mnist_image, success_rate


class NeuralNetworkModel:
    def __init__(self, layers_sizes, batch_size=32, dropout=0.1, init_scale=0.05):
        self._shapes = np.array([layers_sizes[:-1], layers_sizes[1:]]).T
        weights_size = np.sum(np.prod(self._shapes, 1)) + 2*len(self._shapes)
        self._weights = np.random.uniform(-init_scale, init_scale, weights_size)
        self.dropout = dropout
        self.batch_size = batch_size

    def learn(self, data, labels, learning_rate):
        g = grad(self._cost)
        self._weights -= g(self._weights, data, labels) * learning_rate

    def _process_layers(self, weights, data, learning=True):
        for W, mean, std in self._generate_layers(weights):
            data = normalize(data, mean, std)
            data = relu(data)
            if learning and self.dropout is not None:
                data = dropout(data, self.dropout)
            data = np.dot(data, W)
        return data

    def _generate_layers(self, weights):
        used = 0
        for shape in self._shapes:
            size = np.prod(shape)
            yield weights[used:used+size].reshape(shape), weights[used+size], weights[used+size+1] + 1
            used += size + 2

    def answer(self, data):
        return softmax(self._process_layers(self._weights, data, False))

    def _cost(self, weights, data, labels):
        return -np.sum(logsoftmax(self._process_layers(weights, data)) * labels) / self.batch_size

    def epochs_learn(self, training_data, training_labels, transformation=scale_and_rotate_mnist_image, learning_rates=np.linspace(0.1, 0.01, 25)):
        for epoch, learning_rate in enumerate(learning_rates):
            perm = range(len(training_data))
            np.random.shuffle(perm)
            for batch in xrange(len(training_data) / self.batch_size):
                subset = perm[batch * self.batch_size:(batch+1) * self.batch_size]
                data_batch = training_data[subset]
                for i in xrange(self.batch_size):
                    data_batch[i] = transformation(data_batch[i])
                labels_batch = training_labels[subset]
                self.learn(data_batch, labels_batch, learning_rate)
            yield None

    def test(self, data, labels):
        return success_rate(self.answer(data), labels)

    def __repr__(self):
        return 'NeuralNetworkModel:\nlayers = {0}\nbatch_size = {1}\ndropout = {2}'.\
            format(self._shapes[:, 0][1:], self.batch_size, self.dropout)
