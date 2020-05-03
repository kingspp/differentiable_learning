# -*- coding: utf-8 -*-
"""
@created on: 5/3/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

# -*- coding: utf-8 -*-
"""
@created on: 5/3/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import torch.nn as nn
import torch
from torchvision import models

# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
# from pprint import pprint


# pprint(nn.Sequential(*list(model.children())[:-2]))
# exit()
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=True)
# model.eval()
# inp = torch.ones((1, 3, 224,224))
# pprint('------------'*50)
# pprint(list(model.state_dict().keys()))
# print(model.state_dict()['features.denseblock1.denselayer1.conv1.weight'](inp).shape)


# class DenseNetTest(nn.Module):
#     def __init__(self, original_model):
#         super(DenseNetTest, self).__init__()
#         self.features = list(original_model.children())[:-1]
#
#     def forward(self, x):
#         print('input_shape : ', x.shape)
#         # x = self.features(x)
#         for i, layer in enumerate(*self.features):
#             x = layer(x)
#             print(f'Name: {layer}|\nLayer {i + 1} - {x.shape}')
#         exit()
#         return x
#
#
# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
# dense = DenseNetTest(model)
# inp = torch.ones((1, 3, 224,224))
# dense(inp)
# outputs = res50_conv2(inputs)
# outputs.data.shape

# class ResNet50Bottom(nn.Module):
#     def __init__(self, original_model):
#         super(ResNet50Bottom, self).__init__()
#         self.features = nn.Sequential(*list(original_model.children())[:-2])
#         # print(self.features)
#         print(list(original_model.children())[:-2])
#         print('*' * 50)
#
#         for x in list(original_model.children()):
#             print(x)
#             print('-------'*10)
#         exit(())
#
#     def forward(self, x):
#         x = self.features(x)
#         return x
#
#
# res50_model = models.resnet50(pretrained=True)
# res50_conv2 = ResNet50Bottom(res50_model)
#
# outputs = res50_conv2(inputs)
# outputs.data.shape

#
# import torch
#
# from difflr.models import LinearClassifierGSC
# from difflr.data import FashionMNISTDataset
#
#
# def main():
#     torch.manual_seed(0)
#
#     config = {
#         'model_name': 'gsc_ffn',
#         "num_classes": 10,
#         'in_features': 784,
#         'epochs': 10,
#         'batch_size': 256,
#         'lr': 1e-3,
#         'dnn_config':
#             {
#
#                 'layers': [100, 50, 10]
#             }
#     }
#
#     model = LinearClassifierGSC(config=config)
#     model.fit(dataset=FashionMNISTDataset)
#
#
# if __name__ == '__main__':
#     main()
