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


class SimpleCNN(Model):
    def __init__(self, config):
        super().__init__(name='simple_cnn', config=config)
        self.conv_layers = nn.ModuleList([])
        self.linear_layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.Softmax(dim=-1)

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
