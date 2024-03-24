import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications.vgg16 import VGG16 as VGG16_base
from tensorflow.keras.applications.vgg19 import VGG19 as VGG19_base
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications import resnet_rs





# class CNN(Model):
#     def __init__(self, layers, env):
#         super().__init__()
#         self.env = env

#         self.model_layers = [self._build_layer(layer) for layer in layers]
#         # # run sample data through model to initialize
#         input_data = np.random.random((1, *env.observation_space.shape))
#         input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
#         _ = self(input_tensor)

#     def _build_layer(self, layer_config):    
#         layer_type = layer_config['type']
#         if layer_type == 'conv':
#             layer = Conv2D(**layer_config['params'])
#         elif layer_type == 'pool':
#             layer = MaxPooling2D(**layer_config['params'])
#         elif layer_type == 'dropout':
#             layer = Dropout(**layer_config['params'])
#         elif layer_type == 'batchnorm':
#             layer = BatchNormalization(**layer_config['params'])
#         else:
#             raise ValueError(f"Unsupported layer type: {layer_type}")
            
#         return layer
    

#     def __call__(self, inputs):
#         x = inputs
#         for layer in self.model_layers:
#             x = layer(x)
#         # x = Flatten()(x) # will be flattened by model.py models
#         return x

class CNN(Model):
    def __init__(self, layers, env):
        super().__init__()
        self.env = env
        self.input_layer = Input(shape=env.observation_space.shape)
        x = self.input_layer
        for layer in layers:
            x = self._build_layer(layer)(x)
        self.output_layer = x
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        # Call the model with a sample input to build it
        sample_input = np.random.random((1, *self.env.observation_space.shape))
        _ = self(sample_input)

    
    def _build_layer(self, layer_config):
        layer_type = layer_config['type']
        # print(f'layer type: {layer_type}')
        # if layer_type == 'input':
        #     layer = Input(**layer_config['params'])
        if layer_type == 'conv':
            layer = Conv2D(**layer_config['params'])
        elif layer_type == 'pool':
            layer = MaxPooling2D(**layer_config['params'])
        elif layer_type == 'dropout':
            layer = Dropout(**layer_config['params'])
        elif layer_type == 'batchnorm':
            layer = BatchNormalization(**layer_config['params'])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        return layer

    def call(self, x):
        return self.model(x)
    
class ResNet50():
    def __init__(self, input_shape, pooling):
        self.model = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, pooling=pooling)

    def __call__(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNet101():
    def __init__(self, input_shape, pooling):
        self.model = resnet_v2.ResNet101V2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNet152():
    def __init__(self, input_shape, pooling):
        self.model = resnet_v2.ResNet152V2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS50(Model):
    def __init__(self, input_shape, pooling):
        super().__init__()
        self.model = resnet_rs.ResNetRS50(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS101():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS101(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS152():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS152(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS200():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS200(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS270():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS270(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS350():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS350(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class ResNetRS420():
    def __init__(self, input_shape, pooling):
        self.model = resnet_rs.ResNetRS420(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class VGG16():
    def __init__(self, input_shape, pooling):
        self.model = VGG16_base(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def __call__(self, inputs):
        x = self.model(inputs)
        return x
    
class VGG19():
    def __init__(self, input_shape, pooling):
        self.model = VGG19_base(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x