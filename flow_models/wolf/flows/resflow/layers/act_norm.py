import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['ActNorm1d', 'ActNorm2d']


class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        # if actnorm_initialize:
        #     self.register_buffer('initialized', torch.tensor(1))
        # else:
        #     self.register_buffer('initialized', torch.tensor(0))
        self.register_buffer('initialized', torch.tensor(1))
        nn.init.uniform_(self.weight, -1e-5, 1e-5)
        nn.init.uniform_(self.bias, -1e-5, 1e-5)

    @property
    def shape(self):
        raise NotImplementedError


    def forward(self, x, logpx=None, h=None):
        c = x.size(1)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)
        #else:
        #    self.bias.data.copy_(torch.zeros(self.bias.shape[0], device=x.device))
        #    self.weight.data.copy_(torch.zeros(self.weight.shape[0], device=x.device))

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None, h=None):
        # if self.initialized:
        #     self.bias.data.copy_(torch.zeros(self.bias.shape[0], device=y.device))
        #     self.weight.data.copy_(torch.zeros(self.weight.shape[0], device=y.device))
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
