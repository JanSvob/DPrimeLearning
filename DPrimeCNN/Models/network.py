'''
Network model with training and testing function specifications.

networkCfg['inputDepth'] - input depth dimension (3rd dimension)
networkCfg['inputWidth'] - input input
networkCfg['inputHeight'] - input height
networkCfg['numOutputs'] - number of network outputs

'''

from __future__ import print_function

import sys
import importlib
import theano
import theano.tensor as T
import lasagne

sys.path.append('../../Shared/')
import Utils.backprop as backprop


############################ Architecture loading ########################
def import_from_path(path):
    path = path.split('.')[0]
    # python path needs '.' instead of '/' from standard shells.
    path = path.replace('/', '.')
    module = importlib.import_module(path)
    return module

########################### Loss functions ##############################
def norm2(x):
    return T.sqrt(T.sum(T.sqr(x), axis=1) + 1e-12)


def norm2Sqr(x):
    return T.sum(T.sqr(x), axis=1) + 1e-12


def cosdist(x, y):
    denom = norm2(x) * norm2(y)
    return (x * y).sum(axis=1) / (denom + 1e-3)  # T.dot(x, y) / denom


def dprime_loss(outXPos, outYPos, outXNeg, outYNeg, margin = 5, alpha = 0.5):
    norm2Pos = norm2(outXPos - outYPos)
    norm2Neg = norm2(outXNeg - outYNeg)
    return norm2Pos.std() + norm2Neg.std() + norm2Pos.mean() + T.maximum(0.0, margin - norm2Neg.mean())


def siamese_loss(outXPos, outYPos, outXNeg, outYNeg, margin = 5, alpha = 0.5):
    lossData = (1 - alpha) * norm2Sqr(outXPos - outYPos) + alpha * T.sqr(T.maximum(0.0, margin - norm2(outXNeg - outYNeg)))
    return lossData.mean()

loss_func = {'dprime': dprime_loss, 'siamese': siamese_loss }


############################ Main model definition ########################
def getModel(networkCfg):
    print('Building network...', end='')
    sys.stdout.flush()

    ############################ Model definition ########################
    print(' Loading model {0}... '.format(networkCfg['arch_name']), end='')
    network, layers = import_from_path(networkCfg['arch_name']).getArchitecture(networkCfg)
    #########################################################################
    print('Done.')
    print('Network:')
    print('\tInput dimensions (D x W x H): {0} x {1} x {2}'
          .format(networkCfg['inputDepth'], networkCfg['inputWidth'], networkCfg['inputHeight']))
    print('\tNumber of outputs: {0}'.format(networkCfg['numOutputs']))
    print('Compiling functions ... ', end='')
    sys.stdout.flush()
    ########################### Functions definition ########################
    ### Training
    # Create a loss expression for training
    # For this multi-class problem it is cross-entropy loss
    xPos = T.tensor4('xPos')
    xNeg = T.tensor4('xNeg')
    yPos = T.tensor4('yPos')
    yNeg = T.tensor4('yNeg')
    outXPos = lasagne.layers.get_output(network, inputs=xPos)
    outXNeg = lasagne.layers.get_output(network, inputs=xNeg)
    outYPos = lasagne.layers.get_output(network, inputs=yPos)
    outYNeg = lasagne.layers.get_output(network, inputs=yNeg)

    lossData = loss_func[networkCfg['loss']](outXPos, outYPos, outXNeg, outYNeg, networkCfg['distMargin'], networkCfg['alpha'])
    lossReg = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss = lossData + networkCfg['mu'] * lossReg

    # Parameter update expression for training
    # SGD with Adadelta
    params = lasagne.layers.get_all_params(network, trainable=True)
    params = params[-4:]
    updates = lasagne.updates.adadelta(loss, params, learning_rate=networkCfg['lr'],
                                       rho=0.95, epsilon=1e-3)

    grads = T.grad(loss, params)
    gradsNorm = T.nlinalg.norm(T.concatenate([g.flatten() for g in grads]), 2)

    ### Validation/testing
    # Loss expression, this time deterministic fwd-pass (disabling dropout layers)
    outXPosTest = lasagne.layers.get_output(network, inputs=xPos, deterministic=True)
    outYPosTest = lasagne.layers.get_output(network, inputs=yPos, deterministic=True)
    outXNegTest = lasagne.layers.get_output(network, inputs=xNeg, deterministic=True)
    outYNegTest = lasagne.layers.get_output(network, inputs=yNeg, deterministic=True)

    lossTest = loss_func[networkCfg['loss']](outXPosTest, outYPosTest, outXNegTest, outYNegTest, networkCfg['distMargin'], networkCfg['alpha'])

    distPos = norm2(outXPosTest - outYPosTest)
    distPosMean = distPos.mean()
    distNeg = norm2(outXNegTest - outYNegTest)
    distNegMean = distNeg.mean()

    ### Computing feature vectors
    samples = T.tensor4('samples')
    featureVecs = lasagne.layers.get_output(network, inputs=samples, deterministic=True)

    ### Saliency computation
    # replace nonlinearities in the network
    if networkCfg['saliency']:
        relu = lasagne.nonlinearities.rectify
        relu_layers = [layer for layer in lasagne.layers.get_all_layers(network)
            if getattr(layer, 'nonlinearity', None) is relu]
        modified_relu = backprop.GuidedBackprop(relu)
        #modified_relu = backprop.ZeilerBackprop(relu)
        for layer in relu_layers:
            layer.nonlinearity = modified_relu

        # saliency function
        inp = layers['input'].input_var
        featIdx = T.scalar('featIdx', dtype = 'int32')
        outp = lasagne.layers.get_output(network, deterministic = True)
        maxOutp = outp[0, featIdx]
        saliency = theano.grad(maxOutp.sum(), wrt = inp)

    # Compile training and validation functions
    functions = dict()
    functions['train'] = theano.function([xPos, xNeg, yPos, yNeg], [loss, lossData, lossReg, gradsNorm],
                                         updates=updates,
                                         allow_input_downcast=True)
    functions['test'] = theano.function([xPos, xNeg, yPos, yNeg],
                                        [lossTest, distPosMean, distNegMean, distPos, distNeg],
                                        allow_input_downcast=True)
    functions['features'] = theano.function([samples], featureVecs, allow_input_downcast=True)
    if networkCfg['saliency']:
        functions['saliency'] = theano.function([inp, featIdx], saliency)

    #########################################################################
    print('Done.')
    print('Available functions:')
    for key in functions.keys():
        print('\t\'{0}\''.format(key))

    return network, layers, functions
