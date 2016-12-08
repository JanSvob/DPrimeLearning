'''
Data augmentation functions

'''

import numpy as np
from scipy import misc
import skimage
import skimage.transform

defaultCropSize = 8

'''
Helper functions
'''
def frange(start, stop, step):
    i = start
    while i <= stop:
        yield i
        i += step


def cropVec(images, cropSize=defaultCropSize):
    newImages = []
    for idx in range(len(images)):
        newImages.append(crop(images[idx], cropSize=defaultCropSize))

    return np.array(newImages)


'''
Just cropping the image
'''


def crop(image, cropSize=defaultCropSize):
    sizeX, sizeY = image.shape[1] - cropSize, image.shape[2] - cropSize
    sx, sy = cropSize / 2, cropSize / 2
    newImg = image[:, sy:sizeY + sy, sx:sizeX + sx]
    return newImg


'''
Cropping image with only 2 dimensions
'''


def crop2(image, cropSize=defaultCropSize):
    sizeX, sizeY = image.shape[0] - cropSize, image.shape[1] - cropSize
    sx, sy = cropSize / 2, cropSize / 2
    newImg = image[sy:sizeY + sy, sx:sizeX + sx]
    return newImg


'''
Shifting the image - for translation invariance 
'''


def shift2d(image, cropSize=defaultCropSize, step=2):
    # List of new images
    newImages = list()
    # Resulting image size
    sizeX, sizeY = image.shape[0] - cropSize, image.shape[1] - cropSize
    stop = cropSize
    if step > 1:
        stop = cropSize + 1
    for sy in range(0, stop, step):
        for sx in range(0, stop, step):
            newImages.append(image[sy:sizeY + sy, sx:sizeX + sx])

    return newImages


'''
Shifting the image - for translation invariance 
'''


def shift(image, cropSize=defaultCropSize, step=2):
    # List of new images
    newImages = list()
    # Resulting image size
    sizeX, sizeY = image.shape[1] - cropSize, image.shape[2] - cropSize
    stop = cropSize
    if step > 1:
        stop = cropSize + 1
    for sy in range(0, stop, step):
        for sx in range(0, stop, step):
            newImages.append(image[:, sy:sizeY + sy, sx:sizeX + sx])

    return newImages


'''
Shifting the image in place - generates one randomly shifted image
'''


def shiftInPlaceVec(images, cropSize=defaultCropSize):
    newImages = []
    for idx in range(len(images)):
        newImages.append(shiftInPlace(images[idx], cropSize))

    return np.array(newImages)


def shiftInPlace(image, cropSize=defaultCropSize):
    # Resulting image size
    sizeX, sizeY = image.shape[1] - cropSize, image.shape[2] - cropSize
    sx = np.random.randint(cropSize)
    sy = np.random.randint(cropSize)

    return image[:, sy:sizeY + sy, sx:sizeX + sx]


'''
Flipping the image in place - generates one randomly flipped image
'''


def flipInPlaceVec(images):
    newImages = []
    for idx in range(len(images)):
        newImages.append(flipInPlace(images[idx]))

    return np.array(newImages)


def flipInPlace(image):
    origSizeX, origSizeY = image.shape[1], image.shape[2]

    flippedImages = {
        0: image,  # Original image
        1: np.flipud(image),  # image[:, origSizeY-1:0:-1, :],			# Horizontal flip
        2: np.fliplr(image),  # image[:, :, origSizeX-1:0:-1],			# Vertical flip
        3: np.flipud(np.fliplr(image))  # image[:, origSizeY-1:0:-1, origSizeX-1:0:-1] 	# Hor & Vert flip
    }

    flipType = np.random.randint(4)

    return flippedImages[flipType]


'''
Flipping the image
'''


def flip(image):
    # List of new images
    newImages = list()
    # Flip horizontally, vertically and both
    newImages.append(np.flipud(image))
    newImages.append(np.fliplr(image))
    newImages.append(np.flipud(np.fliplr(image)))

    return newImages


'''
Rotating the image - for rotation invariance 
'''


def rotateInPlaceVec(image, maxAngle=2.0, cropSize=defaultCropSize, step=1.0):
    newImages = []
    for idx in range(len(images)):
        newImages.append(rotateInPlace(images[idx], maxAngle, cropSize, step))

    return np.array(newImages)


def rotateInPlace(image, maxAngle=2.0, cropSize=defaultCropSize, step=1.0):
    origSizeX, origSizeY = image.shape[1], image.shape[2]
    halfCrop = cropSize / 2
    angles = []
    for angle in frange(-maxAngle, maxAngle, step):
        angles.append(angle)

    rotIdx = np.random.randint(len(angles))
    anImage = np.zeros((image.shape[0], image.shape[1] - cropSize, image.shape[2] - cropSize))
    for idx in range(image.shape[0]):
        currImg = misc.imrotate(image[idx, :, :], angles[rotIdx])
        currImg = currImg[halfCrop:origSizeY - halfCrop, halfCrop:origSizeX - halfCrop]
        anImage[idx] = currImg

    return anImage


def rotate(image, maxAngle=2.0, cropSize=defaultCropSize, step=1.0):
    # List of new images
    newImages = list()
    # Original image size
    origSizeX, origSizeY = image.shape[1], image.shape[2]
    halfCrop = cropSize / 2
    for angle in frange(-maxAngle, maxAngle, step):
        if abs(angle) > 1e-5:
            anImage = misc.imrotate(image[0, :, :], angle)
            anImage = anImage.reshape(1, anImage.shape[0], anImage.shape[1])
            anImage = anImage[:, halfCrop:origSizeY - halfCrop, halfCrop:origSizeX - halfCrop]
            newImages.append(anImage)

    return newImages


'''
Random rotation or shift
'''


def rotateOrShiftInPlaceVec(images):
    newImages = []
    for idx in range(len(images)):
        newImages.append(rotateOrShiftInPlace(images[idx]))

    return np.array(newImages)


def rotateOrShiftInPlace(image):
    op = np.random.randint(2)

    if op == 0:
        return shiftInPlace(image)
    elif op == 1:
        return rotateInPlace(image)


def resizeInPlaceVec(images, targetDim):
    newImages = []
    for idx in range(len(images)):
        newImages.append(resizeInPlace(images[idx], targetDim))

    return np.array(newImages)


def resizeInPlace(image, targetDim):
    anImage = misc.imresize(image[0], targetDim)
    return anImage.reshape(1, anImage.shape[0], anImage.shape[1])


def scaleInPlace(image, cropSize=defaultCropSize):
    shapeOrig = image.shape
    img = np.rollaxis(image, 0, 3)
    img = skimage.transform.rescale(img, 1.10, preserve_range=True)
    img = np.rollaxis(img, 2, 0)
    shapeNew = img.shape
    diff = shapeNew[2] - shapeOrig[2] - defaultCropSize
    sx, sy = (np.random.random(2) * diff).astype(np.int32)
    sz = shapeOrig[2] - defaultCropSize

    return img[:, sy:sz + sy, sx:sz + sx]


def scaleInPlaceVec(images):
    newImages = []
    for idx in range(len(images)):
        newImages.append(scaleInPlace(images[idx]))

    return np.array(newImages)


def shiftOrScaleInPlace(image):
    op = np.random.randint(2)

    if op == 0:
        return shiftInPlace(image)
    elif op == 1:
        return scaleInPlace(image)


def shiftOrScaleInPlaceVec(images):
    newImages = []
    for idx in range(len(images)):
        newImages.append(shiftOrScaleInPlace(images[idx]))

    return np.array(newImages)
