'''
Simple CNN architecture consisting of 4 convolutional layers 
followed by max-pooling, dense layer and softmax classifier.

networkCfg['inputDepth'] - input depth dimension (3rd dimension)
networkCfg['inputWidth'] - input input
networkCfg['inputHeight'] - input height
networkCfg['numOutputs'] - number of network outputs

'''

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn

def getArchitecture(networkCfg):
	# Input layer
	l_in = lasagne.layers.InputLayer(shape = (None, 
		networkCfg['inputDepth'], networkCfg['inputWidth'], networkCfg['inputHeight']))
	
	l_conv2d1 = lasagne.layers.dnn.Conv2DDNNLayer(
		l_in, num_filters = 32, filter_size = (9, 9),
		stride = (2, 2),
		nonlinearity = lasagne.nonlinearities.rectify,#lasagne.nonlinearities.ScaledTanH(scale_in = 0.5, scale_out = 2.27),
		W = lasagne.init.GlorotUniform())
	l_bn1 = lasagne.layers.batch_norm(l_conv2d1)

	l_conv2d2 = lasagne.layers.dnn.Conv2DDNNLayer(
		l_bn1, num_filters = 64, filter_size = (7, 7),
		nonlinearity = lasagne.nonlinearities.rectify)#ScaledTanH(scale_in = 0.5, scale_out = 2.27))
	#l_bn2 = lasagne.layers.batch_norm(l_conv2d2)
	l_mp2 = lasagne.layers.Pool2DLayer(l_conv2d2, pool_size = (2, 2),
		mode = 'max')#mode = 'average_exc_pad') 

	l_conv2d3 = lasagne.layers.dnn.Conv2DDNNLayer(
		l_mp2, num_filters = 128, filter_size = (5, 5),
		nonlinearity = lasagne.nonlinearities.ScaledTanH(scale_in = 0.5, scale_out = 2.27))
	l_conv2d4 = lasagne.layers.dnn.Conv2DDNNLayer(
		l_conv2d3, num_filters = 256, filter_size = (3, 3),
		nonlinearity = lasagne.nonlinearities.ScaledTanH(scale_in = 0.5, scale_out = 2.27))

	# A fully-connected layer of numOutputs units with 50% dropout
	l_fc1 = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(l_conv2d4, 0.3),
		num_units = 512,
		nonlinearity = lasagne.nonlinearities.identity)#rectify)#identity)
	network = lasagne.layers.DenseLayer(
	    l_fc1,
	    num_units = networkCfg['numOutputs'],
	    nonlinearity = lasagne.nonlinearities.identity)

	return network, {'input': l_in,'conv1': l_conv2d1, 'conv2': l_conv2d2, 'conv3': l_conv2d3, 'conv4': l_conv2d4 }

