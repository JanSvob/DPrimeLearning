''' 
Backpropagation helpers and modifications.

Inspired by:
https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
'''

from __future__ import print_function

import theano


class ModifiedBackprop(object):
    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memorizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed
        if theano.sandbox.cuda.cuda_enabled:
            try_move_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            try_move_to_gpu = lambda x: x
        # We move the input to GPU if needed
        x = try_move_to_gpu(x)
        # We note the tensor type of the input var. to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed)
            outp = try_move_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression
            op = theano.OpFromGraph([inp], [outp])
            # and replace the gradient with our own (defined in a subclass)
            op.grad = self.grad
            # Finally we memorize the new Op
            self.ops[tensor_type] = op
        # And apply the memorized Op to the input we got
        return self.ops[tensor_type](x)


class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd < 0).astype(dtype),)


class ZeilerBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        # return (grd * (grd > 0).astype(inp.dtype), )
        return (self.nonlinearity(grd),)
