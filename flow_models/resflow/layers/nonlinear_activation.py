import torch
import torch.nn as nn
from torch.nn import Parameter
import math

#__all__ = ['ActNorm1d', 'ActNorm2d']

class Sigmoid(nn.Module):

    def __init__(self, eps=1e-5):
        super(Sigmoid, self).__init__()

    def forward(self, x, logpx=None):
        if logpx is None:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(x), logpx + self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        if logpy is None:
            return torch.log(1. - y) - torch.log(y)
        else:
            return torch.log(1. - y) - torch.log(y), logpy - self._logdetgrad(y)

    def _logdetgrad(self, x):
        return torch.log(torch.exp(-x) / (1. + torch.exp(-x)) ** 2).view(x.shape[0],-1).sum(1).reshape(x.shape[0],1)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))

class Tanh(nn.Module):

    def __init__(self, eps=1e-5):
        super(Tanh, self).__init__()

    def forward(self, x, logpx=None):
        if logpx is None:
            return torch.tanh(x)
        else:
            return torch.tanh(x), logpx + self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        if logpy is None:
            return 0.5 * (torch.log(1. + y) - torch.log(1. - y))
        else:
            return 0.5 * (torch.log(1. + y) - torch.log(1. - y)), logpy - self._logdetgrad(y)

    def _logdetgrad(self, x):
        return torch.log(4. * torch.exp(- 2. * x) / (1. + torch.exp(- 2. * x)) ** 2).view(x.shape[0], -1).sum(1).reshape(x.shape[0],1)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))

# noinspection PyUnusedLocal
class LogitTransform_(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward_transform(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).reshape(x.size(0), -1).sum(1).reshape(x.size(0),1)

    def reverse(self, y, logpy=None, **kwargs):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).reshape(x.size(0), -1).sum(1).reshape(x.size(0),1)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad

    def __repr__(self):
        return '{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__)