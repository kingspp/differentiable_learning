# -*- coding: utf-8 -*-
"""
@created on: 5/3/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from difflr.models import Model
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class SimpleCNN(Model):
    def __init__(self, config):
        super().__init__(name='simple_cnn', config=config)
        self.conv_layers = nn.ModuleList([])
        self.linear_layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.LogSoftmax(dim=-1)

        # Build convolution layers
        for e, maps in enumerate(self.config['dnn_config']["filters_maps"]):
            in_channels = self.config['in_features'][0] if e == 0 else self.config['dnn_config']["filters_maps"][e - 1]
            kernel_size, stride = self.config['dnn_config']['kernel_size'][e], self.config['dnn_config']['stride'][e]
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=maps, kernel_size=kernel_size, stride=stride)
            self.conv_layers.extend([conv_layer])

        # calculate the output shape of final convolution operation dynamically
        flattened_op = self._flatten_conv(self.config['in_features'])

        # Build linear layers
        for e, node in enumerate(self.config['dnn_config']['linear']):
            prev_node = flattened_op if e == 0 else self.config['dnn_config']['linear'][e - 1]
            self.linear_layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        x = x.reshape(-1, *self.config['in_features'])
        for layer in self.conv_layers:
            x = self.relu_activation(layer(x))
        x = x.view(-1, x.shape[1:].numel())
        for layer in self.linear_layers[:-1]:
            x = self.relu_activation(layer(x))
        return self.softmax_activation(self.linear_layers[-1](x))

    def _flatten_conv(self, input_shape):
        bs = 1
        inp = Variable(torch.rand(bs, *input_shape))
        output_feat = self._forward_features(inp)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def evaluate(self):
        pass


class GSCCNN(Model):
    def __init__(self, config):
        super().__init__(name='gsc_cnn', config=config)
        self.conv_layers = nn.ModuleList([])
        self.adaptive_pool_layers = nn.ModuleList([])
        self.linear_layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.Softmax(dim=-1)
        layer_ip_shape = self.config['in_features']
        cum_sum = [self.config['in_features'][0]]

        # Build convolution layers
        for e, maps in enumerate(self.config['dnn_config']["filters_maps"]):

            in_channels = cum_sum[0] if e == 0 else cum_sum[-1]
            cum_sum.append(sum(cum_sum) + maps)

            kernel_size, stride = self.config['dnn_config']['kernel_size'][e], self.config['dnn_config']['stride'][e]
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=maps, kernel_size=kernel_size, stride=stride)
            self.conv_layers.extend([conv_layer])
            if e < len(self.config['dnn_config']["filters_maps"]) - 1:
                layer_ip_shape[-3] = in_channels
                layer = self.conv_layers[-1]
                output_shape = self.calc_op_shape(layer, layer_ip_shape)
                layer_ip_shape = output_shape
                adaptive_pool_layer = nn.AdaptiveAvgPool2d(output_size=output_shape[-2:])
                self.adaptive_pool_layers.extend([adaptive_pool_layer])

        # Dynamically calculate the size of flattened layer
        flattened_op = self._flatten_conv([1, *self.config['in_features']])

        # Build linear layers
        for e, node in enumerate(self.config['dnn_config']['linear']):
            prev_node = flattened_op if e == 0 else self.config['dnn_config']['linear'][e - 1]
            self.linear_layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        def to_concat(inp, pool_layer):
            return [pool_layer(x) for x in inp]

        layer_inputs = []
        x = x.reshape(-1, *self.config['in_features'])
        layer_inputs.append(x)
        for e, layer in enumerate(self.conv_layers):
            out = self.relu_activation(layer(x))
            if e < len(self.config['dnn_config']["filters_maps"]) - 1:
                x = torch.cat((out, *to_concat(layer_inputs, self.adaptive_pool_layers[e])), dim=1)
                layer_inputs.append(x)
        x = out
        x = x.view(-1, x.shape[1:].numel())
        for layer in self.linear_layers[:-1]:
            x = self.relu_activation(layer(x))
        return self.softmax_activation(self.linear_layers[-1](x))

    # def __init__(self, config):
    #     super().__init__(name='simple_cnn', config=config)
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
    #     self.ad1 = nn.AdaptiveAvgPool2d((13, 13))
    #     self.conv2 = nn.Conv2d(in_channels=33, out_channels=64, kernel_size=3, stride=1)
    #     self.ad2 = nn.AdaptiveAvgPool2d((11, 11))
    #     self.conv3 = nn.Conv2d(in_channels=98, out_channels=128, kernel_size=3, stride=1)
    #     self.ad3 = nn.AdaptiveAvgPool2d((9, 9))
    #     self.conv4 = nn.Conv2d(in_channels=260, out_channels=32, kernel_size=3, stride=1)
    #     self.ad4 = nn.AdaptiveAvgPool2d((7, 7))
    #     self.fc1 = nn.Linear(1568, 200)
    #     self.fc2 = nn.Linear(200, 10)
    #
    #     print(self)
    #     # exit()
    #
    # def forward(self, x):
    #     print("x.shape", x.shape)
    #     c1_out = F.relu(self.conv1(x))
    #     print("c1_out.shape", c1_out.shape)
    #     c2_in = torch.cat((c1_out, self.ad1(x)), dim=1)
    #     print("c2_in.shape", c2_in.shape)
    #     c2_out = F.relu(self.conv2(c2_in))
    #     print("c2_out.shape", c2_out.shape)
    #     c3_in = torch.cat((c2_out, self.ad2(x), self.ad2(c2_in)), dim=1)
    #     print("c3_in.shape", c3_in.shape)
    #     c3_out = F.relu(self.conv3(c3_in))
    #     print("c3_out.shape", c3_out.shape)
    #     c4_in = torch.cat((c3_out, self.ad3(x), self.ad3(c2_in), self.ad3(c3_in)), dim=1)
    #     print("c4_in.shape", c4_in.shape)
    #     c4_out = F.relu(self.conv4(c4_in))
    #     print("c4_out.shape", c4_out.shape)
    #     exit()
    #     x = x.view(-1, x.shape[1:].numel())
    #     x = F.relu(self.fc1(x))
    #     return F.log_softmax(self.fc2(x))

    def _flatten_conv(self, input_shape):
        for layer in self.conv_layers:
            input_shape[1] = layer.state_dict()['weight'].size()[1]
            x = self.calc_op_shape(layer, input_shape)
            input_shape = x
        return np.prod(x)

    def evaluate(self):
        pass

    def calc_op_shape(self, layer, data_shape):
        print(layer)
        print(data_shape)
        d = torch.rand(1, *data_shape) if len(data_shape) == 3 else torch.rand(*data_shape)
        return [x for x in layer(d).size()]
