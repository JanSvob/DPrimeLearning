'''
Helper functions for loading datasets.

'''

from __future__ import print_function

import os
import numpy as np
import scipy as sp
import skimage.restoration as restoration

'''
Loading dataset from Hong-Kong polytechnic university provided by Ajay Kumar
'''


def rgb2gray(rgb):
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)


def localNormalize(img, mean, std, eps=0.1):
    meanVals = sp.ndimage.gaussian_filter(img, mean)
    imgCent = img - meanVals
    stdVals = np.sqrt(sp.ndimage.gaussian_filter(imgCent ** 2, std))
    return imgCent / (stdVals + eps)


def loadDatasetHKPolyDB(srcFolder, targetDim=None, normalize=True, toFloat=True, verbose=True,
                        modFactor=2, trainSplit=[0], testSplit=[1]):
    # image list
    images = []
    labels = []
    # get all folders in the current directory
    dirList = os.listdir(srcFolder)
    dirList.sort()
    # for each directory, extract the label and load images
    cnt = 0
    for aDir in dirList:
        # we care only about directories
        sampleDir = os.path.join(srcFolder, aDir)
        if os.path.isdir(sampleDir):
            # extract label number (first 3 letters of the directory name)
            aLabel = int(cnt / 2)  # int(aDir[:3]) - 1
            if verbose:
                print('Current label: {0} of real label {1}'.format(aLabel, int(aDir[:3]) - 1))
            # list directory entries
            dirEntries = os.listdir(sampleDir)
            dirEntries.sort()
            for anEntry in dirEntries:
                entryPath = os.path.join(sampleDir, anEntry)
                # we care only about files, only .bmp
                if os.path.isfile(entryPath) and os.path.splitext(entryPath)[1] == '.bmp':
                    # load image and add to array of images
                    anImage = sp.misc.imread(entryPath)
                    if targetDim != None:
                        anImage = sp.misc.imresize(anImage, targetDim)
                    # Add dimension specifying numbef or channels
                    if toFloat:
                        anImage = anImage / np.float32(256)
                    if normalize:
                        anImage = localNormalize(anImage, 7, 7, eps=1.0)
                    images.append(anImage.reshape(1, anImage.shape[0], anImage.shape[1]))
                    labels.append(aLabel)

                    # print('Append image {0} with label {1}'.format(anEntry, aLabel))
        cnt += 1

    # split data into training and testing subset
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    for idx in range(len(images)):
        # Perform data augmentation
        augImages = list()
        augImages.append(images[idx])
        augLabels = np.empty(len(augImages))
        augLabels.fill(labels[idx])
        augLabels = augLabels.tolist()
        # Add training samples
        if (labels[idx] % modFactor) in trainSplit:
            X_train.extend(augImages)
            y_train.extend(augLabels)
        # Add testing and validation samples
        if (labels[idx] % modFactor) in testSplit:
            X_test.extend(augImages)
            y_test.extend(augLabels)
            X_val.extend(augImages)
            y_val.extend(augLabels)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


'''
Loading PolyU dataset
'''


def loadDatasetPolyU(srcFolder, targetDim=None, verbose=True, toFloat=True, normalize=True):
    # image list
    images = []
    labels = []
    # get all folders in the current directory
    fileList = os.listdir(srcFolder)
    fileList.sort()
    # for each directory, extract the label and load images
    for aFile in fileList:
        filePath = os.path.join(srcFolder, aFile)
        # we care only about files, only .bmp
        if os.path.isfile(filePath) and os.path.splitext(filePath)[1] == '.bmp':
            # skip the label if it is in exclude list
            aLabel = int(aFile[6:9])
            aLabel = aLabel - 1
            # extract label
            if verbose:
                print('Current label: {0}'.format(aLabel))
            # read image and add to array of images
            anImage = sp.misc.imread(filePath)
            if targetDim != None:
                anImage = sp.misc.imresize(anImage, targetDim)
            # Add dimension specifying numbef or channels
            if toFloat:
                anImage = anImage / np.float32(256)
            if normalize:
                anImage = localNormalize(anImage, 5, 15, eps=1.0)
                anImage = restoration.denoise_tv_chambolle(anImage, weight=0.05)
            images.append(anImage.reshape(1, anImage.shape[0], anImage.shape[1]))

            labels.append(aLabel)

    # split data into training and testing subset
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    for idx in range(len(images)):
        # Perform data augmentation
        augImages = list()
        augImages.append(images[idx])
        augLabels = np.empty(len(augImages))
        augLabels.fill(labels[idx])
        augLabels = augLabels.tolist()
        # Add training samples
        if labels[idx] % 2 == 0:
            X_train.extend(augImages)
            y_train.extend(augLabels)
        # Add testing and validation samples
        if labels[idx] % 2 != 0:
            X_test.append(images[idx])
            y_test.append(labels[idx])
            X_val.append(images[idx])
            y_val.append(labels[idx])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


'''
Loading IITD dataset
'''


def loadDatasetIITD(srcFolder, targetDim=None, verbose=True, toFloat=True, normalize=True,
                    modFactor=2, trainSplit=[0], testSplit=[1]):
    # image list
    images = []
    labels = []
    # get all folders in the current directory
    fileList = os.listdir(srcFolder)
    fileList.sort()
    # for each directory, extract the label and load images
    for aFile in fileList:
        filePath = os.path.join(srcFolder, aFile)
        # we care only about files, only .bmp
        if os.path.isfile(filePath) and os.path.splitext(filePath)[1] == '.bmp':
            # skip the label if it is in exclude list
            aLabel = int(aFile[:3])
            aLabel = aLabel - 1
            # extract label
            if verbose:
                print('Current label: {0}'.format(aLabel))
            # read image and add to array of images
            anImage = sp.misc.imread(filePath)
            if targetDim != None:
                anImage = sp.misc.imresize(anImage, targetDim)
            # Add dimension specifying numbef or channels
            if toFloat:
                anImage = anImage / np.float32(256)
            if normalize:
                anImage = localNormalize(anImage, 7, 7, eps=1.0)
            images.append(anImage.reshape(1, anImage.shape[0], anImage.shape[1]))

            labels.append(aLabel)

    # split data into training and testing subset
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    for idx in range(len(images)):
        # Perform data augmentation
        augImages = list()
        augImages.append(images[idx])
        augLabels = np.empty(len(augImages))
        augLabels.fill(labels[idx])
        augLabels = augLabels.tolist()
        # Add training samples
        if (labels[idx] % modFactor) in trainSplit:
            augImages = list()
            augImages.append(images[idx])
            augLabels = [labels[idx]]
            X_train.extend(augImages)
            y_train.extend(augLabels)
        # Add testing and validation samples
        if (labels[idx] % modFactor) in testSplit:
            X_test.extend(augImages)
            y_test.extend(augLabels)
            X_val.extend(augImages)
            y_val.extend(augLabels)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


'''
Loading Casia dataset
'''


def loadDatasetCasia(srcFolder, targetDim=None, verbose=True, normalize=True, toFloat=True, modFactor=2,
                     trainSplit=[0], testSplit=[1]):
    # image list
    images = []
    labels = []
    # get all folders in the current directory
    fileList = os.listdir(srcFolder)
    fileList.sort()
    # for each directory, extract the label and load images
    for aFile in fileList:
        filePath = os.path.join(srcFolder, aFile)
        # we care only about files, only .jpg
        if os.path.isfile(filePath) and os.path.splitext(filePath)[1] == '.jpg':
            # extract label
            aLabel = int(aFile[:3]) - 1
            if verbose:
                print('Current label: {0}'.format(aLabel))
            # read image and add to array of images
            anImage = sp.misc.imread(filePath)
            if targetDim != None:
                anImage = sp.misc.imresize(anImage, targetDim)
            # Add dimension specifying numbef or channels
            if toFloat:
                anImage = anImage / np.float32(256)
            if normalize:
                anImage = localNormalize(anImage, 7, 7, eps=0.1)
            images.append(anImage.reshape(1, anImage.shape[0], anImage.shape[1]))

            labels.append(aLabel)

    labels = np.array(labels)
    # split data into training and testing subset
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    for idx in range(len(images)):
        # Perform data augmentation
        augImages = list()
        augImages.append(images[idx])
        augLabels = np.empty(len(augImages))
        augLabels.fill(labels[idx])
        augLabels = augLabels.tolist()
        # Add training samples
        if (labels[idx] % modFactor) in trainSplit:
            X_train.extend(augImages)
            y_train.extend(augLabels)
        # Add testing and validation samples
        if (labels[idx] % modFactor) in testSplit:
            X_test.append(images[idx])
            y_test.append(labels[idx])
            X_val.append(images[idx])
            y_val.append(labels[idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test
