import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, h=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x, h=h)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx, h=h)
            return x, logpx

    def inverse(self, y, logpy=None, h=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y, h=h)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy, h=h)
            return y, logpy


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None, h=None):
        return self.flow.inverse(x, logpx, h=h)

    def inverse(self, y, logpy=None, h=None):
        return self.flow.forward(y, logpy, h=h)
