import torch as T
import torch.nn as nn


def VarianceScaling_(tensor: T.tensor, scale: float=1.0, mode: str='fan_in', distribution: str='normal'):

    ##DEBUG
    print(f'scale: {scale}')
    print(f'mode: {mode}')
    print(f'distribution: {distribution}')

    # dimensions = tensor.dim()
    # if dimensions < 2:
    #     raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    
    if mode == 'fan_in':
        fan = tensor.size(0)
    elif mode == 'fan_out':
        fan = tensor.size(1)
    elif mode == 'fan_avg':
        fan = (tensor.size(0) + tensor.size(1)) / 2
    else:
        raise ValueError("Mode {} not supported, please use 'fan_in', 'fan_out', or 'fan_avg'.".format(mode))

    val = 1.0 / T.sqrt(T.tensor(fan))

    if distribution == 'normal' or distribution == 'truncated_normal':   
        if distribution == 'normal':
            with T.no_grad():
                nn.init.normal_(tensor, 0, val)
                tensor.mul_(scale)
        elif distribution == 'truncated_normal':
            with T.no_grad():
                nn.init.trunc_normal_(tensor, mean=0.0, std=val, a=-2.0, b=2.0)
                tensor.mul_(scale)
    elif distribution == 'uniform':
        with T.no_grad():
            tensor.uniform_(-val, val)
            tensor.mul_(scale)
    else:
        raise ValueError("Distribution {} not supported, please use 'normal', 'truncated_normal', or 'uniform'.".format(distribution))