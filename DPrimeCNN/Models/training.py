'''
Training functions

'''

from __future__ import print_function

import time
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
import sys

sys.path.append('../../Shared/')
import Utils.augmentation as dataAugmentation

'''
Iterating over the samples generating training/testing batches for
multiclass classification (inputs, targets).
'''


def iterateBatchesClassif(inputs, targets, batchSize, shuffle=False):
    # there has to be as many inputs as targets
    assert len(inputs) == len(targets)

    # if required, shuffle indices
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # generate batches and yield inputs and targets
    for startIdx in range(0, len(inputs) - batchSize + 1, batchSize):
        if shuffle:
            selection = indices[startIdx:startIdx + batchSize]
        else:
            selection = slice(startIdx, startIdx + batchSize)

        # Yield the output
        yield inputs[selection], targets[selection]


'''
Iterating over the samples generating training/testing batches of
triplets for siamese training and distance computation (positive, postitive2, negative).
'''


def iterateBatchesTriplet(data, labels, batchSize, shuffle=False, maxSamples=0):
    # there has to be as many data samples as labels
    assert len(data) == len(labels)

    # Create index array
    indices = np.arange(len(data))
    # From now on work on indices
    uniqueLabels = list(set(labels))
    numClasses = len(uniqueLabels)
    posIndices1 = []
    posIndices2 = []
    negIndices = []
    # Divide into classes
    for lbl in range(0, numClasses - 1):
        classIndices = indices[labels == uniqueLabels[lbl]]
        nonClassIndices = indices[labels != uniqueLabels[lbl]]
        # If required, suffle class and non class indices
        if shuffle:
            np.random.shuffle(classIndices)
            np.random.shuffle(nonClassIndices)

        # All pairwise combinations of positive
        classPairs = np.array(list(itertools.combinations(classIndices, 2)))
        for neg in nonClassIndices:
            if shuffle:
                np.random.shuffle(classPairs)
            for pair in classPairs:
                posIndices1.append(pair[0])
                posIndices2.append(pair[1])
                negIndices.append(neg)

    # generate batches and yield
    # if required, shuffle data
    finalIndices = np.arange(len(posIndices1))
    if shuffle:
        np.random.shuffle(finalIndices)
    posIndices1 = np.array(posIndices1)
    posIndices2 = np.array(posIndices2)
    negIndices = np.array(negIndices)
    if maxSamples <= 0:
        print('Taking all {0} samples to generate {1}-sample batches.'.format(len(posIndices1), batchSize))
        lastStartIdx = len(posIndices1) - batchSize + 1
    elif maxSamples > 0 and maxSamples < batchSize:
        print('Taking {0} samples from {1} to generate {2}-sample batches.'.format(batchSize, len(posIndices1),
                                                                                   batchSize))
        lastStartIdx = 1
    else:
        print('Taking {0} samples from {1} to generate {2}-sample batches.'.format(maxSamples, len(posIndices1),
                                                                                   batchSize))
        lastStartIdx = maxSamples - batchSize + 1
    sys.stdout.flush()
    for startIdx in range(0, lastStartIdx, batchSize):
        sel = finalIndices[startIdx:startIdx + batchSize]
        selection1 = posIndices1[sel]
        selection2 = posIndices2[sel]
        selection3 = negIndices[sel]

        augDataPos = []
        augDataPos2 = []
        augDataNeg = []
        if not shuffle:
            augDataPos = dataAugmentation.cropVec(data[selection1])
            augDataPos2 = dataAugmentation.cropVec(data[selection2])
            augDataNeg = dataAugmentation.cropVec(data[selection3])
        else:
            augDataPos = data[selection1]
            augDataPos2 = data[selection2]
            augDataNeg = data[selection3]
            augDataPos = dataAugmentation.shiftOrScaleInPlaceVec(augDataPos)
            augDataPos2 = dataAugmentation.shiftOrScaleInPlaceVec(augDataPos2)
            augDataNeg = dataAugmentation.shiftOrScaleInPlaceVec(augDataNeg)

        yield augDataPos, augDataPos2, augDataNeg


'''
Basic training function.

functions['train'] - training function
functions['test'] - validation function

trainingCfg['numEpochs'] - number of training epochs
trainingCfg['batchSize'] - size of batches used during trainig

trainData['data'] - training data
trainData['labels'] - training labels

valData['data'] - validation data
valData['labels'] - validation labels
'''


def train(functions, trainingCfg, trainData, valData):
    print('Starting training ... ')
    sys.stdout.flush()

    assert 'batchSize' in trainingCfg, 'batchSize training configuration missing.'
    assert 'numEpochs' in trainingCfg, 'numEpochs training configuration missing.'
    assert 'maxSamples' in trainingCfg, 'maxSamples training configuration missing.'

    assert 'train' in functions, 'No training function.'
    assert 'test' in functions, 'No validation function.'

    assert 'data' in trainData, 'No training data.'
    assert 'labels' in trainData, 'No training labels.'
    assert 'data' in valData, 'No validation data.'
    assert 'labels' in valData, 'No validations labels.'

    X_train = trainData['data']
    y_train = trainData['labels']

    X_val = valData['data']
    y_val = valData['labels']

    # Run all epochs
    for epoch in range(trainingCfg['numEpochs']):
        # Pass over the training data
        trainError = 0
        trainDataError = 0
        trainRegError = 0
        trainGradNorm = 0
        numTrainBatches = 0
        startTime = time.time()
        for batch in iterateBatchesTriplet(X_train, y_train, trainingCfg['batchSize'], shuffle=True,
                                           maxSamples=trainingCfg['maxSamples']):
            pos1, pos2, neg = batch
            [err, dataErr, regErr, gradNorm] = functions['train'](pos1, pos1, pos2, neg)
            trainError += err
            trainDataError += dataErr
            trainRegError += regErr
            trainGradNorm += gradNorm
            numTrainBatches += 1

        # Pass over the validation data
        valError = 0
        distancePosMean = 0
        distanceNegMean = 0
        distancesPos = []
        distancesNeg = []
        numValBatches = 0
        for batch in iterateBatchesTriplet(X_val, y_val, trainingCfg['batchSize'], shuffle=False,
                                           maxSamples=trainingCfg['maxSamples']):
            pos1, pos2, neg = batch
            error, distPosMean, distNegMean, distPos, distNeg = functions['test'](pos1, pos1, pos2, neg)
            valError += error
            distancePosMean += distPosMean
            distanceNegMean += distNegMean
            distancesPos.extend(distPos.tolist())
            distancesNeg.extend(distNeg.tolist())
            numValBatches += 1

        yScore = np.array(distancesPos + distancesNeg)
        yTrue = np.concatenate((np.ones(len(distancesPos)), np.zeros(len(distancesNeg))))
        rocAuc = roc_auc_score(yTrue, -yScore)

        # Results of the currect epoch
        print('Epoch {} of {} took {:.3f}s'.format(epoch + 1,
                                                   trainingCfg['numEpochs'], time.time() - startTime))
        print('\tTraining loss:\t\t{:.6f} \tdata:{:.6f} \treg:{:.6f} \tgrad:{:.6f}'.format(trainError / numTrainBatches,
                                                                                           trainDataError / numTrainBatches,
                                                                                           trainRegError / numTrainBatches,
                                                                                           trainGradNorm / numTrainBatches))
        print('\tValidation loss:\t{:.6f}'.format(valError / numValBatches))
        print('\tValidation distance positive:\t{:.6f}'.format(distancePosMean / numValBatches))
        print('\tValidation distance negative:\t{:.6f}'.format(distanceNegMean / numValBatches))
        print('\tROC area under curve:\t{:.4f}'.format(rocAuc))
        sys.stdout.flush()

        yield trainError / numTrainBatches, valError / numValBatches, rocAuc

    print('Done.')
