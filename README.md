# Dual Path Networks in Keras
[Dual Path Networks](https://arxiv.org/abs/1707.01629) are highly efficient networks which combine the strength of both ResNeXt [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) and DenseNets [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).

Note: Weights have not been ported over yet.

## Dual Path Connections
<img src="https://github.com/titu1994/Keras-DualPathNetworks/blob/master/images/dual%20path%20networks.png?raw=true" width="100%" height="100%">

## Usage
Several of the standard Dual Path Network models have been included. They can be initialized as : 
```
from dual_path_network import DPN92, DPN98, DPN107, DPN137

model = DPN92(input_shape=(224, 224, 3)) # same for the others
```

To create a custom DualPathNetwork, use the DualPathNetwork builder method : 
```
from dual_path_network import DualPathNetwork

model = DualPathNetwork(input_shape=(224, 224, 3),
                        initial_conv_filters=64,
                        depth=[3, 4, 20, 3],
                        filter_increment=[16, 32, 24, 128],
                        cardinality=32,
                        width=3,
                        weight_decay=0,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=1000)
```

## Performance
<img src="https://github.com/titu1994/Keras-DualPathNetworks/blob/master/images/original-results-on-imagenet1k.png?raw=true" height=100% width=100%>

## Support 
- Keras does not have inbuilt support for grouped convolutions. Therefore I had to use lambda layers to match the ResNeXt paper implementation. When grouped convolution support is added, I hope to add it in this as well.
- Mean-Max Global Pooling support is present with the help of Lambda layer to scale the sum.
- Depth and Filter_Increment must be lists for now, and must be lists of same length. Will think about adding support for integers, but I think list support is far more useful anyway, so I may not implement it.
- Weight decay support is added, but disabled by default. The DPN paper does not mention it, but ResNet, WRN and ResNeXt paper may all use small weight regularization. Use a small value of `1e-4` or `5e-4` if you wish to use it.
