import sys
sys.path.append('../Shared/')
import Utils.datasets as datasets
import Utils.snapshotter as snapshotter
import Utils.augmentation as dataAugmentation
import Utils.tests as tests
import argparse
import Utils.command_line_test as command_line_test
import Models.network as network
import lasagne
import os
import sys

#splitsTrain = [[0], [1], [1, 2], [0, 3], [1, 2, 3], [0, 4, 5], [0, 2, 3], [1, 4, 5]]
#splitsTest = [[1], [0], [0, 3], [1, 2], [0, 4, 5], [1, 2, 3], [1, 4, 5], [0, 2, 3]]
#splitsMod = [2, 2, 4, 4, 6, 6, 6, 6]

def main(argv):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Learning palmprint recognition using CNNs.')
    args, networkCfg, testingCfg = command_line_test.parse(parser)

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

    # Create results dir if doesn't exist
    if not os.path.exists(testingCfg['outputDir']):
        os.makedirs(testingCfg['outputDir'])

    # Load configurations
    print('Loading settings ...')
    persistentDb = snapshotter.Snapshotter(testingCfg['model'] + '/model.hdf5')
    targetDim = (136, 136)
    networkCfg = persistentDb.load('config_network')
    #testingCfg = persistentDb.load('config_testing')

    print('Load params at epoch: {0}'.format(testingCfg['epoch']))
    modelParamsKey = 'Epoch_%05i_params' % testingCfg['epoch']
    modelParams = persistentDb.load(modelParamsKey)

    # Load model parameters
    print('Setting network parameters ...')
    lasagne.layers.set_all_param_values(cnnetwork, modelParams)

    # Generate leave-one-out tests
    testData = dict()
    testData['data'] = dataAugmentation.cropVec(X_test)
    testData['labels'] = y_test
    loaTests = tests.genLeaveOneOut(functions['features'], testingCfg, testData)
    cntr = 0
    problematic = []
    for output in tests.testLeaveOneOut3(loaTests):
        cntr += 1
        # Compute number of errors
        print('-----------------')
        print('Probe sample: {0}'.format(cntr))
        scoresPos, scoresNeg, results = output
        numErrors = 0
        for res in results:
            if res[1] != res[2]:
                numErrors += 1
        #print('\tNumber of scores: {0} pos, {1} neg'.format(len(scoresPos), len(scoresNeg)))
        print('\tNumber of errors: {0}/{1}'.format(numErrors, len(scoresPos)))
        # Save pos and neg scores
        filePos = open(
            os.path.join(testingCfg['outputDir'], 'genuine_' + testingCfg['dataset'] + '_' + str(cntr) + '.csv'),
            'w')
        filePos.write(str(scoresPos[0]))
        for score in scoresPos[1:]:
            filePos.write(', ' + str(score))
        filePos.write('\n')
        filePos.close()
        fileNeg = open(
            os.path.join(testingCfg['outputDir'], 'impostor_' + testingCfg['dataset'] + '_' + str(cntr) + '.csv'),
            'w')
        fileNeg.write(str(scoresNeg[0]))
        for score in scoresNeg[1:]:
            fileNeg.write(', ' + str(score))
        fileNeg.write('\n')
        fileNeg.close()

    persistentDb.close()


if __name__ == '__main__':
    main(sys.argv)
