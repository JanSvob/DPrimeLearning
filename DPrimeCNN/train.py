#!/usr/bin/env python

"""
Palmprint recognition using CNNs.
"""

from __future__ import print_function

import sys
import os

import argparse

import numpy as np
import lasagne

sys.path.append('../Shared/')
import Utils.datasets as datasets
import Models.network as network
import Models.training as training
import Utils.snapshotter as snapshotter
import Utils.command_line as command_line


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(argv):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Learning palmprint recognition using CNNs.')
    args, networkCfg, trainingCfg, testingCfg = command_line.parse(parser)

    # Load the dataset
    print("Loading data...")
    if args.dataset == 'HKPoly':
        X_train, y_train, X_val, y_val, X_test, y_test = datasets.loadDatasetHKPolyDB('../Datasets/HKPoly/2D/',
                                                                                      targetDim=(
                                                                                          args.dim_width,
                                                                                          args.dim_height),
                                                                                      normalize=args.norm_input,
                                                                                      modFactor=args.split_mod,
                                                                                      trainSplit=args.split_train,
                                                                                      testSplit=args.split_test)
    elif args.dataset == 'IITD':
        X_train, y_train, X_val, y_val, X_test, y_test = datasets.loadDatasetIITD('../Datasets/IITD/2D/',
                                                                                  targetDim=(
                                                                                      args.dim_width, args.dim_height),
                                                                                  normalize=args.norm_input,
                                                                                  modFactor=args.split_mod,
                                                                                  trainSplit=args.split_train,
                                                                                  testSplit=args.split_test)
    elif args.dataset == 'Casia':
        X_train, y_train, X_val, y_val, X_test, y_test = datasets.loadDatasetCasia(
            '../Datasets/Casia/2D_Filt/Right_Filt/',
            targetDim=(args.dim_width, args.dim_height), normalize=args.norm_input, modFactor=args.split_mod,
            trainSplit=args.split_train, testSplit=args.split_test)
    else:
        print('Dataset {0} is not available'.format(args.dataset))
        return

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_val.shape)
    print(y_val.shape)

    # Create neural network model (depending on first command line parameter)
    cnnetwork, cnnlayers, functions = network.getModel(networkCfg)

    # Prepare the model output directory
    if not os.path.exists(args.model):
        os.makedirs(args.model)

    # Run the training
    modelFileName = 'model'
    if args.oper == 'train' or args.oper == 'retrain':
        # Save network configuraiton to the database
        persistentDb = snapshotter.Snapshotter(os.path.join(args.model, modelFileName + '.hdf5'))

        # If the model is being retrained
        if args.oper == 'retrain':
            # Load model parameters from file
            modelParamsKey = 'Epoch_%05i_params' % 1
            if trainingCfg['startEpoch'] > 0:
                modelParamsKey = 'Epoch_%05i_params' % trainingCfg['startEpoch']
            param_values = persistentDb.load(modelParamsKey)
            lasagne.layers.set_all_param_values(cnnetwork, param_values)

        persistentDb.store('network_inputDepth', networkCfg['inputDepth'])
        persistentDb.store('network_inputWidth', networkCfg['inputWidth'])
        persistentDb.store('network_inputHeight', networkCfg['inputHeight'])
        persistentDb.store('network_numOutputs', networkCfg['numOutputs'])
        persistentDb.store('training_numEpochs', args.num_epochs + args.start_epoch)
        persistentDb.store('training_batchSize', args.batch_size)
        persistentDb.store('config_network', networkCfg)
        persistentDb.store('config_training', trainingCfg)
        persistentDb.store('config_testing', testingCfg)

        trainData = dict()
        trainData['data'] = X_train
        trainData['labels'] = y_train
        valData = dict()
        valData['data'] = X_val
        valData['labels'] = y_val

        # After each epoch, loss values are yield
        epochCntr = trainingCfg['startEpoch']
        try:
            for epochData in training.train(functions, trainingCfg, trainData, valData):
                epochCntr += 1
                trainLoss, valLoss, rocAuc = epochData
                persistentDb.store('Epoch_%05i_losstrain' % epochCntr, trainLoss)
                persistentDb.store('Epoch_%05i_lossval' % epochCntr, valLoss)
                persistentDb.store('Epoch_%05i_aucval' % epochCntr, rocAuc)
                persistentDb.store('Epoch_%05i_params' % epochCntr, lasagne.layers.get_all_param_values(cnnetwork))

            # Save the trained model to file
            persistentDb.close()
            np.savez(os.path.join(args.model, modelFileName + '.npz'), *lasagne.layers.get_all_param_values(cnnetwork))
        except KeyboardInterrupt:
            print('Training interrupted. Closing databse.')
            persistentDb.close()
            np.savez(os.path.join(args.model, modelFileName + '.npz'), *lasagne.layers.get_all_param_values(cnnetwork))
    elif args.oper == 'load':
        # Load pre-trained model from file
        with np.load(os.path.join(args.model, modelFileName + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(cnnetwork, param_values)
    else:
        print('Operation {0} is not available.'.format(args.oper))
        return


if __name__ == '__main__':
    '''
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print('Trains a neural network for palmprint recognition.')
        print('Usage: {0} [METHOD [FILENAME]]'.format(sys.argv[0]))
        print()
        print('METHOD: whether to use training or load model from file')
        print('        \'train\' - train the model')
        print('        \'load\' - load model from file')
        print('	       \'retrain\' - retrain the model given initial parameters')
        print('FILENAME: model file name. If the model is trained, it')
        print('          is saved into this file. If the model is loaded or')
        print('          retrained, it is loaded from this file.')
    else:
    '''
    main(sys.argv)
