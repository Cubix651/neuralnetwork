# Author: Jakub Cislo
# http://cislo.net.pl
# jakub@cislo.net.pl
# License: MIT
# Copyright (C) 2016

import gzip
import autograd.numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from autograd.scipy.misc import logsumexp
import cPickle


def read_int32(f):
    return np.fromstring(f.read(4), dtype=np.dtype('>i4'))[0]


def read_images(path):
    with gzip.open(path) as f:
        magic = read_int32(f)
        if magic != 2051:
            raise ValueError('bad magic number')
        images_count = read_int32(f)
        rows = read_int32(f)
        cols = read_int32(f)
        images = np.fromstring(f.read(), dtype=np.uint8)
        images.resize((images_count, cols * rows))
        return images / 255.


def read_labels(path):
    with gzip.open(path) as f:
        magic = read_int32(f)
        if magic != 2049:
            raise ValueError('bad magic number')
        labels_count = read_int32(f)
        labels = np.fromstring(f.read(), dtype=np.uint8)
        vectorized_labels = np.zeros((labels_count, 10))
        for no, label in enumerate(labels):
            vectorized_labels[no][label] = 1.
        return vectorized_labels

def read_cifar_set(path):
    with open(path, 'rb') as f:
        d = cPickle.load(f)
        vectorized_labels = np.zeros((len(d['labels']), 10))
        for no, label in enumerate(d['labels']):
            vectorized_labels[no][label] = 1.
        return d['data'], vectorized_labels


def plt_image(img):
    plt.imshow(1. - img, cmap='gray', interpolation='nearest', vmin=0., vmax=1.)


def plt_image_color(img):
    plt.imshow(img, interpolation='nearest')


def scale_and_rotate_mnist_image(image, angle_range=15.0, scale_range=0.1):
        angle = 2 * angle_range * np.random.random() - angle_range
        scale = 1 + 2 * scale_range * np.random.random() - scale_range

        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
        tf_scale = transform.SimilarityTransform(scale=scale)
        tf_shift = transform.SimilarityTransform(translation=[-14, -14])
        tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])

        image = transform.warp(image.reshape([28, 28]),
                               (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
        return image.reshape([28*28])


def softmax(v):
    exp = np.exp(v)
    return exp / np.sum(exp, 1).reshape(-1, 1)


def logsoftmax(v):
    return v - logsumexp(v, 1).reshape(-1, 1)


def relu(v):
    return np.maximum(v, 0)


def dropout(v, drp):
    return (np.random.uniform(size=v.shape) > drp) * v / (1 - drp)


def normalize(v, mean, std):
    standard = (v - np.mean(v, 1).reshape(-1, 1)) / np.std(v, 1).reshape(-1, 1)
    return standard * std + mean


def success_rate(output, expected_output):
    a = np.argmax(output, 1)
    b = np.argmax(expected_output, 1)
    return float(np.sum(a == b)) / len(expected_output)
