import os,sys
import numpy as np

import find_mxnet
import mxnet as mx
import mxnet.ndarray as nd

import mxnet.autograd as autograd

def _group_norm_func(input_data, num_groups, eps, gamma, beta):
    n,c,h,w = input_data.shape

    input_data = nd.reshape(data=input_data, shape=(n, num_groups, c/num_groups, h, w))

    # mean
    mean = nd.mean(input_data, axis=2, keepdims=True)

    # std
    temp = nd.square((input_data-mean))
    std = nd.sqrt(nd.sum(temp, axis=2, keepdims=True) / (c/num_groups))

    input_data = (input_data-mean) / (std+eps)
    out = input_data.reshape((n,c,h,w))
    gamma = gamma.reshape((c,1,1))
    beta = beta.reshape((c,1,1))
    out = out*gamma + beta
    return out

class GroupNormalization(mx.operator.CustomOp):
    def __init__(self, num_groups=32, eps=1e-10):
        self.num_groups = long(num_groups)
        self.eps = long(eps)

    def forward(self, is_train, req, in_data, out_data, aux):
        input_data = in_data[0]
        assert self.num_groups < input_data.shape[1]
        gamma = in_data[1]
        beta = in_data[2]
        out = _group_norm_func(input_data, self.num_groups, self.eps, gamma, beta)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        input_data = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        out = out_data[0]

        input_data.attach_grad()
        gamma.attach_grad()
        beta.attach_grad()

        with autograd.record():
            out = _group_norm_func(input_data, self.num_groups, self.eps, gamma, beta)
        out.backward(out_grad[0])

        self.assign(in_grad[0], req[0], input_data.grad)
        self.assign(in_grad[1], req[0], gamma.grad)
        self.assign(in_grad[2], req[0], beta.grad)

@mx.operator.register('GroupNormalization')
class GroupNormalizationProp(mx.operator.CustomOpProp):
    def __init__(self, num_groups=32, eps=1e-5):
        super(GroupNormalizationProp, self).__init__()
        self.num_groups = long(num_groups)
        self.eps = long(eps)

    def list_arguments(self):
        return ['data', 'gamma', 'beta']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        in_shape = in_shape[0]
        c = in_shape[1]
        return [in_shape, (c,), (c,)], [in_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return GroupNormalization(self.num_groups, self.eps)
