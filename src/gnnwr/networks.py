import math
import torch
import torch.nn as nn


def default_dense_layer(insize, outsize):
    """
    generate default dense layers for neural network

    Parameters
    ----------
    insize: int
        input size of neural network
    outsize: int
        output size of neural network

    Returns
    -------
    dense_layer: list
        a list of dense layers of neural network
    """
    dense_layer = []
    size = int(math.pow(2, int(math.log2(insize))))
    while size > outsize:
        dense_layer.append(size)
        size = int(math.pow(2, int(math.log2(size)) - 1))
    return dense_layer

class LinearNetwork(nn.Module):
    """
    LinearNetwork is a neural network with dense layers, which is used to calculate the weight of features.
    | The each layer of LinearNetwork is as follows:
    | full connection layer -> batch normalization layer -> activate function -> drop out layer

    Parameters
    ----------
    dense_layer: list
        a list of dense layers of Neural Network
    insize: int
        input size of Neural Network(must be positive)
    outsize: int
        Output size of Neural Network(must be positive)
    drop_out: float
        drop out rate(default: ``0.2``)
    activate_func: torch.nn.functional
        activate function(default: ``nn.PReLU(init=0.1)``)
    batch_norm: bool
        whether use batch normalization(default: ``True``)
    """
    def __init__(self, insize, outsize, drop_out=0, activate_func=None, batch_norm=False):
        super(LinearNetwork, self).__init__()
        self.layer = nn.Linear(insize, outsize)
        if drop_out < 0 or drop_out > 1:
            raise ValueError("drop_out must be in [0, 1]")
        elif drop_out == 0:
            self.drop_out = nn.Identity()
        else:
            self.drop_out = nn.Dropout(drop_out)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(outsize)
        else:
            self.batch_norm = nn.Identity()
        
        if activate_func is None:
            self.activate_func = nn.Identity()
        else:
            self.activate_func = activate_func
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.kaiming_uniform_(self.layer.weight, a=0, mode='fan_in')
        if self.layer.bias is not None:
            self.layer.bias.data.fill_(0)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer(x)
        x = self.batch_norm(x)
        x = self.activate_func(x)
        x = self.drop_out(x)
        return x
    
    def __str__(self) -> str:
        return f"LinearNetwork: {self.layer.in_features} -> {self.layer.out_features}\n" + \
                f"Dropout: {self.drop_out.p}\n" + \
                f"BatchNorm: {self.batch_norm}\n" + \
                f"Activation: {self.activate_func}"
    
    def __repr__(self) -> str:
        return self.__str__()

class DistanceProjection(nn.Module):
    """
    DistanceProjection projects a high-dimensional distance vector to a fixed
    embedding dimension, decoupling knn_k from SWNN hidden layer structure.

    Parameters
    ----------
    insize: int
        input size (distance vector dimension, e.g. knn_k)
    embed_dim: int
        output embedding dimension (default: ``128``)
    drop_out: float
        drop out rate (default: ``0.1``)
    """
    def __init__(self, insize, embed_dim=128, drop_out=0.1):
        super(DistanceProjection, self).__init__()
        self.linear = nn.Linear(insize, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.activate = nn.PReLU(init=0.1)
        if drop_out > 0:
            self.dropout = nn.Dropout(drop_out)
        else:
            self.dropout = nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=0, mode='fan_in')
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.dropout(x)
        return x


class SWNN(nn.Module):
    """
    SWNN is a neural network with dense layers, which is used to calculate the spatial and temporal weight of features.
    | The each layer of SWNN is as follows:
    | full connection layer -> batch normalization layer -> activate function -> drop out layer

    Parameters
    ----------
    dense_layer: list
        a list of dense layers of Neural Network
    insize: int
        input size of Neural Network(must be positive)
    outsize: int
        Output size of Neural Network(must be positive)
    drop_out: float
        drop out rate(default: ``0.2``)
    activate_func: torch.nn.functional
        activate function(default: ``nn.PReLU(init=0.1)``)
    batch_norm: bool
        whether use batch normalization(default: ``True``)
    """
    def __init__(self, dense_layer=None, insize=-1, outsize=-1, drop_out=0.2, activate_func=nn.PReLU(init=0.1),
                 batch_norm=True, embed_dim=None):

        super(SWNN, self).__init__()
        if insize < 0 or outsize < 0:
            raise ValueError("insize and outsize must be positive")

        self.insize = insize
        self.outsize = outsize
        self.drop_out = drop_out
        self.batch_norm = batch_norm
        self.activate_func = activate_func

        # Distance projection: project high-dim distance vectors to fixed embed_dim
        if embed_dim is not None and insize > embed_dim:
            self.projection = DistanceProjection(insize, embed_dim)
            effective_insize = embed_dim
        else:
            self.projection = None
            effective_insize = insize

        if dense_layer is None or len(dense_layer) == 0:
            self.dense_layer = default_dense_layer(effective_insize, outsize)
        else:
            self.dense_layer = dense_layer

        count = 0  # used to name layers
        lastsize = effective_insize  # used to record the size of last layer
        self.fc = nn.Sequential()

        for size in self.dense_layer:
            # add full connection layer
            self.fc.add_module("swnn_full" + str(count),
                               LinearNetwork(lastsize, size, drop_out, activate_func, batch_norm))
            lastsize = size
            count += 1
        self.fc.add_module("full" + str(count),
                            LinearNetwork(lastsize, self.outsize))

    def forward(self, x):
        x = x.to(torch.float32)
        if self.projection is not None:
            x = self.projection(x)
        x = self.fc(x)
        return x


class STPNN(nn.Module):
    """
    STPNN is a neural network with dense layers, which is used to calculate the spatial and temporal proximity
    of two nodes.
    | The each layer of STPNN is as follows:
    | full connection layer -> batch normalization layer -> activate function -> drop out layer

    Parameters
    ----------
    dense_layer: list
        a list of dense layers of Neural Network
    insize: int
        input size of Neural Network(must be positive)
    outsize: int
        Output size of Neural Network(must be positive)
    drop_out: float
        drop out rate(default: ``0.2``)
    activate_func: torch.nn.functional
        activate function(default: ``nn.ReLU()``)
    batch_norm: bool
        whether use batch normalization(default: ``False``)
    """
    def __init__(self, dense_layer, insize, outsize, drop_out=0.2, activate_func=nn.ReLU(), batch_norm=False):

        super(STPNN, self).__init__()
        # default dense layer
        self.dense_layer = dense_layer
        self.drop_out = drop_out
        self.batch_norm = batch_norm
        self.activate_func = activate_func
        self.insize = insize
        self.outsize = outsize
        count = 0  # used to name layers
        lastsize = self.insize  # used to record the size of last layer
        self.fc = nn.Sequential()
        for size in self.dense_layer:
            self.fc.add_module("stpnn_full" + str(count),
                                 LinearNetwork(lastsize, size, drop_out, activate_func, batch_norm))
            lastsize = size
            count += 1
        self.fc.add_module("full" + str(count),
                            LinearNetwork(lastsize, self.outsize,activate_func=activate_func))

    def forward(self, x):
        # STPNN
        x = x.to(torch.float32)
        batch = x.shape[0]
        height = x.shape[1]
        x = torch.reshape(x, shape=(batch * height, x.shape[2]))
        output = self.fc(x)
        output = torch.reshape(output, shape=(batch, height * self.outsize))
        return output


class STNN_SPNN(nn.Module):
    """
    STNN_SPNN is a neural network with dense layers, which is used to calculate the spatial proximity of two nodes
    and temporal proximity of two nodes at the same time.
    | The each layer of STNN and SPNN is as follows:
    | full connection layer -> activate function

    Parameters
    ----------
    STNN_insize: int
        input size of STNN(must be positive)
    STNN_outsize: int
        Output size of STNN(must be positive)
    SPNN_insize: int
        input size of SPNN(must be positive)
    SPNN_outsize: int
        Output size of SPNN(must be positive)
    activate_func: torch.nn.functional
        activate function(default: ``nn.ReLU()``)

    """
    def __init__(self, STNN_insize:int, STNN_outsize, SPNN_insize:int, SPNN_outsize, activate_func=nn.ReLU()):

        super(STNN_SPNN, self).__init__()
        self.STNN_insize = STNN_insize
        self.STNN_outsize = STNN_outsize
        self.SPNN_insize = SPNN_insize
        self.SPNN_outsize = SPNN_outsize
        self.activate_func = activate_func
        self.STNN = nn.Sequential(nn.Linear(self.STNN_insize, self.STNN_outsize), self.activate_func)
        self.SPNN = nn.Sequential(nn.Linear(self.SPNN_insize, self.SPNN_outsize), self.activate_func)

    def forward(self, input1):
        STNN_input = input1[:, :, self.SPNN_insize:]
        SPNN_input = input1[:, :, 0:self.SPNN_insize]
        STNN_output = self.STNN(STNN_input)
        SPNN_output = self.SPNN(SPNN_input)
        output = torch.cat((STNN_output, SPNN_output), dim=-1)
        return output