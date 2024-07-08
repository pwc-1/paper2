import numpy as np
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
import ipdb

class TemporalEncoder(nn.Cell):
    def __init__(self, input_dim, dropout):
        super(TemporalEncoder, self).__init__()
        dim_feedforward = 256
        self.conv = Aggregationmodel([[1,3,4],[3,3,4],[3,3,10],[3,3,20]], input_dim, dims=[input_dim])
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.self_atten = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear1 = nn.Dense(input_dim, dim_feedforward, has_bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, input_dim, has_bias=True)
        self.norm1 = nn.LayerNorm(normalized_shape=(input_dim,))
        self.norm2 = nn.LayerNorm(normalized_shape=(input_dim,))
        self.dropout0 = nn.Dropout(p=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def construct(self, features):
        src = features
        src1 = self.conv(src)
        src = src + self.dropout0(src1)
        src = src.permute(2, 0, 1)
        src2 = self.self_atten(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(ops.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(1, 2, 0)
        return src


class MSTALayer(nn.Cell):
    def __init__(self, input_channels, out_channels, kernel_sizes, unit_sizes, fusion_type='add'):
        super(MSTALayer, self).__init__()
        assert len(unit_sizes) == len(kernel_sizes), 'unit_sizes and kernel_sizes should have the same length'
        self.unit_sizes = unit_sizes
        self.layer_0 = nn.Conv2d(input_channels, out_channels, kernel_sizes[0], stride=1, pad_mode='same', padding=0, group=4, has_bias=True)
        self.layer_1 = nn.Conv2d(input_channels, out_channels, kernel_sizes[1], stride=1, pad_mode='same', padding=0, group=4, has_bias=True)
        self.layer_2 = nn.Conv2d(input_channels, out_channels, kernel_sizes[2], stride=1, pad_mode='same', padding=0, group=4, has_bias=True)
        self.layer_3 = nn.Conv2d(input_channels, out_channels, kernel_sizes[3], stride=1, pad_mode='same', padding=0, group=4, has_bias=True)

    
    def construct(self, x):
        shape = x.shape
        out = self.layer_0(x.reshape(shape[0], shape[1], shape[2]//self.unit_sizes[0], self.unit_sizes[0])).reshape(shape)
        out += self.layer_0(x.reshape(shape[0], shape[1], shape[2]//self.unit_sizes[1], self.unit_sizes[1])).reshape(shape)
        out += self.layer_0(x.reshape(shape[0], shape[1], shape[2]//self.unit_sizes[2], self.unit_sizes[2])).reshape(shape)
        out += self.layer_0(x.reshape(shape[0], shape[1], shape[2]//self.unit_sizes[3], self.unit_sizes[3])).reshape(shape)
        return out



class Aggregationmodel(nn.Cell):
    '''Multi-scale Temporal Aggregation (MSTA) Subnet, composed of sequential MSTA layer'''
    def __init__(self, kernels, input_dim, dims=[384, 512]):
        super(Aggregationmodel, self).__init__()
        self.unit_sizes = [x[-1] for x in kernels]
        self.kernel_sizes = [tuple(x[:2]) for x in kernels]
        layers = []
        self.dims = dims
        for i in range(len(self.dims)):
            in_channels = input_dim if i == 0 else self.dims[i-1]
            out_channels = self.dims[i]
            layers += [MSTALayer(in_channels, out_channels, self.kernel_sizes, self.unit_sizes), nn.ReLU()]
        self.layers = nn.SequentialCell(*layers)

    def construct(self, X):
        '''input: (N,C,T)'''
        out_conv = self.layers(X)
        
        return out_conv
