import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, layers, env):
        super(CNN, self).__init__()
        self.layer_config = layers
        self.env = env

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Instantiate empty ModuleDict to hold CNN layers
        self.layers = nn.ModuleDict()

        # set the initial input size
        input_size = self.env.observation_space.shape[-1]
        
        # Build the layers
        for i, layer in enumerate(layers):
            for layer_type, params in layer.items():
                self.layers[f'{layer_type}_{i}'] = self._build_layer(layer_type, params, input_size)
                if layer_type == 'conv':
                    input_size = params['out_channels']

        # Add flatten layer to the end
        self.layers['flatten'] = nn.Flatten()

        # Move the model to the specified device
        self.to(self.device)

    def _build_layer(self, layer_type, params, input_size):
        # set the input size in the params dict
        
        if layer_type == 'conv':
            params['in_channels'] = input_size
            return nn.Conv2d(**params)
        elif layer_type == 'pool':
            return nn.MaxPool2d(**params)
        elif layer_type == 'dropout':
            return nn.Dropout(**params)
        elif layer_type == 'batchnorm':
            params['num_features'] = input_size
            return nn.BatchNorm2d(**params)
        elif layer_type == 'relu':
            return nn.ReLU()
        elif layer_type == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    def get_config(self):
        return {
            'layers': self.layer_config,
            'env': self.env.spec.id,
        }
    
# class ResNet50():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, pooling=pooling)

#     def __call__(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNet101():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_v2.ResNet101V2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNet152():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_v2.ResNet152V2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS50(Model):
#     def __init__(self, input_shape, pooling):
#         super().__init__()
#         self.model = resnet_rs.ResNetRS50(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS101():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS101(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS152():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS152(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS200():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS200(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS270():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS270(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS350():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS350(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class ResNetRS420():
#     def __init__(self, input_shape, pooling):
#         self.model = resnet_rs.ResNetRS420(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class VGG16():
#     def __init__(self, input_shape, pooling):
#         self.model = VGG16_base(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def __call__(self, inputs):
#         x = self.model(inputs)
#         return x
    
# class VGG19():
#     def __init__(self, input_shape, pooling):
#         self.model = VGG19_base(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

#     def call(self, inputs):
#         x = self.model(inputs)
#         return x