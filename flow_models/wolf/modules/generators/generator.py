__author__ = 'max'

import math
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from flow_models.wolf.flows.flow import Flow


class Generator(nn.Module):
    """
    class for Generator with a Flow.
    """

    def __init__(self, flow: Flow):
        super(Generator, self).__init__()
        self.flow = flow

    def add_config(self, config):
        self.config = config

    def sync(self):
        self.flow.sync()

    def generate(self, epsilon: torch.Tensor,
                 h: Union[None, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            epsilon: Tensor [batch, channels, height, width]
                epslion for generation
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor1, Tensor2
            Tensor1: generated tensor [batch, channels, height, width]
            Tensor2: log probabilities [batch]

        """
        # [batch, channel, height, width]
        z, logdet = self.flow.fwdpass(epsilon, h)
        return z, logdet

    def encode(self, x: torch.Tensor, h: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            x: Tensor [batch, channels, height, width]
                The input data.
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor [batch, channels, height, width]
            The tensor for encoded epsilon.

        """
        return self.flow.bwdpass(x, h)[0]

    def log_probability(self, x: torch.Tensor, h: Union[None, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor [batch, channel, height, width]
                The input data.
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor1, Tensor2 [batch,]
            Tensor1: generated tensor [batch, channels, height, width]
            Tensor2: The tensor of the log probabilities of x [batch]

        """
        # [batch, channel, height, width]
        epsilon_org, logdet = self.flow.bwdpass(x, h)
        if self.config.model.name == 'None':
            # [batch, numels]
            epsilon = epsilon_org.view(epsilon_org.size(0), -1)
            # [batch]
            log_probs = epsilon.mul(epsilon).sum(dim=1) + math.log(math.pi * 2.) * epsilon.size(1)
            return epsilon_org, log_probs.mul(-0.5) + logdet
        else:
            return epsilon_org, logdet

    def init(self, data: torch.Tensor, h=None, init_scale=1.0):
        return self.flow.bwdpass(data, h, init=True, init_scale=init_scale)

    @classmethod
    def from_params(cls, params: Dict, config=None) -> "Generator":
        flow_params = params.pop('flow')
        flow_type = flow_params.pop('type')
        if flow_type == 'resflow':
            from flow_models.wolf.flows.resflow import ResidualFlow
            if config.flow.squeeze:
                input_shape = (config.training.batch_size, config.data.num_channels * 4, config.data.image_size // 2,
                               config.data.image_size // 2)
            else:
                input_shape = (
                config.training.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
            flow = ResidualFlow(config, input_shape,
                                      n_blocks=list(map(int, config.flow.nblocks.split('-'))),
                                      intermediate_dim=config.flow.intermediate_dim,
                                      vnorms='ffff',
                                      actnorm=config.flow.actnorm,
                                      grad_in_forward=config.flow.grad_in_forward,
                                      activation_fn=config.flow.act_fn).to(config.device)
        else:
            flow = Flow.by_name(flow_type).from_params(flow_params)
        return Generator(flow)
