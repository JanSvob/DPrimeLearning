#!/usr/bin/env python

from __future__ import print_function

import sys
import numpy as np
import itertools
import augmentation as dataAugmentation

'''
Helper function to iterate over batches.
Supports shuffling of data to generate different batches.
'''


def iterateBatches(inputs, targets, batchSize, shuffle=False):
    # there has to be as many inputs as targets
    assert len(inputs) == len(targets)

    # if required, shuffle indices
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # generate batches and yield inputs and targets
    lastStart = 0
    for startIdx in range(0, len(inputs) - batchSize + 1, batchSize):
        lastStart = startIdx
        if shuffle:
            selection = indices[startIdx:startIdx + batchSize]
        else:
            selection = slice(startIdx, startIdx + batchSize)

        # Yield the output
        yield inputs[selection], targets[selection]

    # send also the last uncompete batch if there's one
    remaining = len(inputs) - (lastStart + batchSize)
    if remaining > 0:
        start = lastStart + batchSize + 1
        selection = slice(start, start + remaining)
        yield inputs[selection], targets[selection]


'''
Compute distance between 2 feature vectors.
'''


def distance(fv1, fv2):
    return dist(fv1, fv2)


def dist(fv1, fv2):
    return np.sqrt(np.sum(np.square(fv1 - fv2)))


def norm2(x):
    return np.sqrt(np.sum(np.square(x)) + 1e-6)


def cosDistance(x, y):
    denom = norm2(x) * norm2(y)
    return np.dot(x, y) / (denom + 1e-6)  # T.dot(x, y) / denom


'''
Testing function that compares each sample to all the others in the database.
Sample is classified in the class with lowest score returned.
'''


def compareAllToAll(fvFunction, testingCfg, testData):
    print('Starting testing all x all ...')
    sys.stdout.flush()

    assert 'batchSize' in testingCfg, 'batchSize testing configuration missing.'

    assert 'data' in testData, 'No testing data.'
    assert 'labels' in testData, 'No testing labels.'

    X_test = testData['data']
    y_test = testData['labels']

    # Prepare arrays of fvs and labels
    featureVecs = []
    labels = []

    # Compute FV for all the samples
    for batch in iterateBatches(X_test, y_test, testingCfg['batchSize']):
        inputs, targets = batch
        fvs = fvFunction(inputs)
        featureVecs.extend(fvs)
        labels.extend(targets)

    # For each sample in the set, comapre with all other samples and return
    # label for the class with the smallest distance
    numSamples = len(featureVecs)
    results = []
    negScores = []
    posScores = []
    for idx in range(numSamples):
        distances = []
        distLabels = []
        for idx2 in range(numSamples):
            if idx != idx2:
                dist = distance(featureVecs[idx], featureVecs[idx2])
                distances.append(dist)
                distLabels.append((labels[idx], labels[idx2]))
                if labels[idx] == labels[idx2]:
                    posScores.append(dist)
                else:
                    negScores.append(dist)

        # Find the minimum and store the result
        minIdx = np.argmin(distances)
        results.append((distances[minIdx], distLabels[minIdx][0], distLabels[minIdx][1]))

    return results, posScores, negScores


def genLeaveOneOut(fvFunction, testingCfg, testData):
    print('Generating leave-one-out tests ...')
    sys.stdout.flush()

    assert 'data' in testData, 'No testing data.'
    assert 'labels' in testData, 'No testing labels.'

    X_test = testData['data']
    y_test = testData['labels']

    # Prepare arrays of fvs and labels
    featureVecs = []
    labels = []

    # Compute FV for all the samples
    featureVecs = fvFunction(X_test)
    labels = y_test

    featureVecs = np.array(featureVecs)
    labels = np.array(labels)
    # For each class, select i-th sample as the probe, leave others as reference
    # There are 5 samples for each class
    numClassSamples = 5  # 5
    uniqueLabels = list(set(labels))
    tests = []
    for i in range(numClassSamples):
        testRun = {}
        testRun['probeFvs'] = []
        testRun['probeLbls'] = []
        testRun['refFvs'] = []
        testRun['refLbls'] = []
        for lbl in uniqueLabels:
            classFvs = featureVecs[labels == lbl]
            classLbls = labels[labels == lbl]
            if i > len(classFvs) - 1:
                break
            testRun['probeFvs'].append(classFvs[i])
            testRun['probeLbls'].append(classLbls[i])
            testRun['refFvs'].append(classFvs[np.arange(len(classFvs)) != i])
            testRun['refLbls'].append(classLbls[np.arange(len(classLbls)) != i])
        testRun['probeFvs'] = np.array(testRun['probeFvs'])
        testRun['probeLbls'] = np.array(testRun['probeLbls'])
        testRun['refFvs'] = np.array(testRun['refFvs'])
        testRun['refLbls'] = np.array(testRun['refLbls'])
        tests.append(testRun)

    return tests


def testLeaveOneOut(tests):
    print('Performing leave-one-out tests ...')
    sys.stdout.flush()

    for i in range(len(tests)):
        # Get current test data
        testData = tests[i]

        negScores = []
        posScores = []
        results = []
        # Loop through all the probe and find shortest distance between all the reference
        for probeIdx in range(len(testData['probeFvs'])):
            probeFv = testData['probeFvs'][probeIdx]
            probeLbl = testData['probeLbls'][probeIdx]
            distances = []
            distLabels = []
            for refIndices in range(len(testData['refFvs'])):
                for refIdx in range(len(testData['refFvs'][refIndices])):
                    refFv = testData['refFvs'][refIndices][refIdx]
                    refLbl = testData['refLbls'][refIndices][refIdx]
                    dist = distance(probeFv, refFv)
                    distances.append(dist)
                    distLabels.append((probeLbl, refLbl))
                    if probeLbl == refLbl:
                        posScores.append(dist)
                    else:
                        negScores.append(dist)

            # Find the minimum and store the result
            minIdx = np.argmin(distances)
            results.append((distances[minIdx], distLabels[minIdx][0], distLabels[minIdx][1]))

        yield posScores, negScores, results


def testLeaveOneOut3(tests):
    print('Performing leave-one-out tests ...')
    sys.stdout.flush()

    for i in range(len(tests)):
        # Get current test data
        testData = tests[i]

        negScores = []
        posScores = []
        results = []
        # Loop through all the probe and find shortest distance between all the reference
        for probeIdx in range(len(testData['probeFvs'])):
            probeFv = testData['probeFvs'][probeIdx]
            probeLbl = testData['probeLbls'][probeIdx]
            distances = []
            distLabels = []
            for refIndices in range(len(testData['refFvs'])):
                dists = []
                refLbl = -5
                for refIdx in range(len(testData['refFvs'][refIndices])):
                    refFv = testData['refFvs'][refIndices][refIdx]
                    refLbl = testData['refLbls'][refIndices][refIdx]
                    dist = distance(probeFv, refFv)
                    dists.append(dist)

                minDist = np.min(dists)
                distances.append(minDist)
                distLabels.append((probeLbl, refLbl))
                if probeLbl == refLbl:
                    posScores.append(minDist)
                else:
                    negScores.append(minDist)

            # Find the minimum and store the result
            minIdx = np.argmin(distances)
            results.append((distances[minIdx], distLabels[minIdx][0], distLabels[minIdx][1]))

        yield posScores, negScores, results


def genLeaveOneOut2(testData, testLabels):
    print('Generating leave-one-out tests ...')
    sys.stdout.flush()

    # Prepare arrays of fvs and labels
    featureVecs = testData
    labels = testLabels

    featureVecs = np.array(featureVecs)
    labels = np.array(labels)
    # For each class, select i-th sample as the probe, leave others as reference
    # There are 5 samples for each class
    numClassSamples = 4
    uniqueLabels = list(set(labels))
    tests = []
    for i in range(numClassSamples):
        testRun = {}
        testRun['probeFvs'] = []
        testRun['probeLbls'] = []
        testRun['refFvs'] = []
        testRun['refLbls'] = []
        for lbl in uniqueLabels:
            classFvs = featureVecs[labels == lbl]
            classLbls = labels[labels == lbl]
            testRun['probeFvs'].append(classFvs[i])
            testRun['probeLbls'].append(classLbls[i])
            testRun['refFvs'].append(classFvs[np.arange(len(classFvs)) != i])
            testRun['refLbls'].append(classLbls[np.arange(len(classLbls)) != i])

        testRun['probeFvs'] = np.array(testRun['probeFvs'])
        testRun['probeLbls'] = np.array(testRun['probeLbls'])
        testRun['refFvs'] = np.array(testRun['refFvs'])
        testRun['refLbls'] = np.array(testRun['refLbls'])
        tests.append(testRun)

    return tests


def testLeaveOneOut2(tests, distf):
    print('Performing leave-one-out tests ...')
    sys.stdout.flush()

    for i in range(len(tests)):
        print('Testing round {0} out of {1}.'.format(i, len(tests)))
        sys.stdout.flush()
        # Get current test data
        testData = tests[i]

        negScores = []
        posScores = []
        results = []
        # Loop through all the probe and find shortest distance between all the reference
        for probeIdx in range(len(testData['probeFvs'])):
            print('Testing probe {0} out of {1}.'.format(probeIdx, len(testData['probeFvs'])))
            sys.stdout.flush()
            probeFv = testData['probeFvs'][probeIdx]
            probeLbl = testData['probeLbls'][probeIdx]
            distances = []
            distLabels = []
            for refIndices in range(len(testData['refFvs'])):
                dists = []
                refLbl = -5
                for refIdx in range(len(testData['refFvs'][refIndices])):
                    refFv = testData['refFvs'][refIndices][refIdx]
                    refLbl = testData['refLbls'][refIndices][refIdx]
                    dist = distf(probeFv[0], probeFv[1], refFv[0], refFv[1])
                    dists.append(dist)

                minDist = np.min(dists)
                distances.append(minDist)
                distLabels.append((probeLbl, refLbl))
                if probeLbl == refLbl:
                    posScores.append(minDist)
                else:
                    negScores.append(minDist)

            # Find the minimum and store the result
            minIdx = np.argmin(distances)
            results.append((distances[minIdx], distLabels[minIdx][0], distLabels[minIdx][1]))
            print('dist = {0}, {1} == {2}'.format(distances[minIdx], distLabels[minIdx][0], distLabels[minIdx][1]))
            sys.stdout.flush()

        yield posScores, negScores, results


def compareAllCouples(scoreFunction, testingCfg, testData):
    print('Comparing all couples ...')
    sys.stdout.flush()

    assert 'data' in testData, 'No testing data.'
    assert 'labels' in testData, 'No testing labels.'

    X_test = testData['data']
    y_test = testData['labels']

    # Prepare arrays of fvs and labels
    scores = []
    labels = []

    # Generate all possible couples and label them
    indices = np.arange(len(X_test))
    # All pairwise combinations of indices
    classPairs = list(itertools.combinations(indices, 2))
    cntr = 0
    for pair in classPairs:
        cntr += 1
        print('Generating {0} of {1}'.format(cntr, len(classPairs)))
        idx1, idx2 = pair[0], pair[1]
        if idx1 != idx2:
            couple = np.array([X_test[idx1], X_test[idx2]])
            score = scoreFunction(np.array([couple.reshape(2, couple.shape[2], couple.shape[3])]))
            scores.append(score[0][0])
            if y_test[idx1] == y_test[idx2]:
                labels.append(1)
            else:
                labels.append(0)

    pairs = np.array(classPairs)
    scores = np.array(scores)
    labels = np.array(labels)

    return pairs, scores, labels
