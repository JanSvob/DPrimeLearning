#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as sp
import lasagne

import Utils.visualization as visualization
import Utils.snapshotter as snapshotter
import Models.network as network


def plot_saliency_map(modelDbName, imgName):
    # Load configurations
    persistentDb = snapshotter.Snapshotter(modelDbName)
    targetDim = (persistentDb.load('network_inputWidth'),
                 persistentDb.load('network_inputHeight'))
    networkCfg = persistentDb.load('config_network')
    modelParamsKey = 'Epoch_%05i_params' % persistentDb.load('training_numEpochs')
    modelParams = persistentDb.load(modelParamsKey)
    persistentDb.close()

    # Load image
    print('Loading image sample ...')
    img = sp.misc.imread(imgName)
    img = sp.misc.imresize(img, targetDim)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])

    # Get network model
    print('Creating network ...')
    cnnetwork, cnnlayers, functions = network.getModel(networkCfg)
    # Load model parameters
    lasagne.layers.set_all_param_values(cnnetwork, modelParams)

    # Compute saliency map
    print('Plottting saliency maps for each feature ...')
    totsal = np.zeros(shape=targetDim)
    for i in range(networkCfg['numOutputs']):
        saliency = functions['saliency'](img, i)
        totsal = totsal + saliency[0][0]
        visualization.plot_saliency_map(img, saliency, i)
    totsal = totsal.reshape(1, 1, totsal.shape[0], totsal.shape[1])
    visualization.plot_saliency_map(img, totsal, networkCfg['numOutputs'])
    print('Done.')
