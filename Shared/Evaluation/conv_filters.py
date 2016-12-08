#!/usr/bin/env python

from __future__ import print_function

import sys
sys.path.append('../DPrimeCNN/')
import Utils.visualization as visualization
import Utils.snapshotter as snapshotter
import Models.network as network
import lasagne

def plot_conv_filters(modelDbName, layerName, epoch=-1):
    # Load configurations
    persistentDb = snapshotter.Snapshotter(modelDbName, readOnly=True)
    networkCfg = persistentDb.load('config_network')
    if epoch == -1:
        epoch = persistentDb.load('training_numEpochs')
    modelParamsKey = 'Epoch_%05i_params' % epoch
    modelParams = persistentDb.load(modelParamsKey)
    persistentDb.close()

    # Create model, load parameters
    print('Creating network ...')
    cnnetwork, cnnlayers, functions = network.getModel(networkCfg)
    # Load model parameters
    lasagne.layers.set_all_param_values(cnnetwork, modelParams)

    # Plot filters
    print('Plotting conv layer filters ...')
    visualization.plot_conv_weights(cnnlayers[layerName], layerName)
