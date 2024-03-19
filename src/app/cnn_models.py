import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import resnet_v2, resnet_rs
from tensorflow.keras.applications.vgg16 import VGG16 as VGG16_base
from tensorflow.keras.applications.vgg19 import VGG19 as VGG19_base
from tensorflow.keras.applications.inception_v3 import InceptionV3




class CNN(Model):
    def __init__(self, input_shape, output_shape, layers):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = self._build_layers(layers)

    def _build_layers(self, layer_configs):
        layers = []
        for layer_config in layer_configs:
            layer_type = layer_config['type']
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
            layers.append(layer)
        return layers
    

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        # x = Flatten()(x) # will be flattened by model.py models
        return x
    
class ResNet50():
    def __init__(self, input_shape, pooling):
        self.model = resnet_v2.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
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

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
class VGG19():
    def __init__(self, input_shape, pooling):
        self.model = VGG19_base(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

    def call(self, inputs):
        x = self.model(inputs)
        return x