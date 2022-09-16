from flow_models import ResidualFlow
from flow_models.wolf import WolfCore
import json
from flow_models.resflow.layers.squeeze import SqueezeLayer
import torch

def flow_forward(config, flow_model, x, log_det=0, reverse=False):
    if config.flow.model == 'resflow':
        if config.flow.squeeze:
            x = SqueezeLayer(2).forward(x)
        if log_det == 0:
            if reverse == False:
                z, neg_log_jacob = flow_model(x, log_det, reverse=reverse)
                if '-' in config.flow.nblocks:
                    z = z.view(x.shape[0], x.shape[1], 2, 2, x.shape[2] // 2, x.shape[3] // 2)
                    z = z.permute(0, 1, 4, 2, 5, 3).reshape(x.shape)
                else:
                    z = z.view(x.shape)
            else:
                if '-' in config.flow.nblocks:
                    x = x.view(x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2, 2, 2).permute(0,1,5,2,3,4)\
                                           .reshape(x.shape[0], x.shape[1], x.shape[2], 2, x.shape[3]//2).permute(0,1,3,2,4).reshape(x.shape)
                z, neg_log_jacob = flow_model(x, log_det, reverse=reverse)
            if config.flow.squeeze:
                z = SqueezeLayer(2).inverse(z)
            return z, - neg_log_jacob.view(x.shape[0])
        else:
            if reverse == False:
                z = flow_model(x, log_det, reverse=reverse)
                if '-' in config.flow.nblocks:
                    z = z.view(x.shape[0], x.shape[1], 2, 2, x.shape[2] // 2, x.shape[3] // 2)
                    z = z.permute(0, 1, 4, 2, 5, 3).reshape(x.shape)
                else:
                    z = z.view(x.shape)
            else:
                if '-' in config.flow.nblocks:
                    x = x.view(x.shape[0], x.shape[1], x.shape[2] // 2, x.shape[3] // 2, 2, 2).permute(0, 1, 5, 2, 3, 4) \
                        .reshape(x.shape[0], x.shape[1], x.shape[2], 2, x.shape[3] // 2).permute(0, 1, 3, 2, 4).reshape(x.shape)
                # z = flow_model(x, log_det, reverse=reverse)
                z = flow_model(x, log_det, reverse=reverse).view(x.shape)
            if config.flow.squeeze:
                z = SqueezeLayer(2).inverse(z)
            return z, -1
    elif config.flow.model == 'glow_v2':
        if not reverse:
            _, logdet, z_out = flow_model(x)
            z_out = flow_model.z_outs_concat(z_out)
        else:
            x = flow_model.x_split(x)
            z_out = flow_model.reverse(x)
            logdet = -1
        return z_out, logdet
    elif config.flow.model == 'wolf':
        if config.flow.squeeze:
            x = SqueezeLayer(2).forward(x)
        if not reverse:
            if log_det == 0:
                z, logdet_kl = flow_model(x, y=None, n_bits=config.flow.n_bits, nsamples=config.flow.train_k, reverse=reverse, eval_logdet=True)
            else:
                z = flow_model(x, y=None, n_bits=config.flow.n_bits, nsamples=config.flow.train_k, reverse=reverse, eval_logdet=False)
                logdet_kl = -1
        else:
            z = flow_model(x, reverse=reverse).view(x.shape)
            logdet_kl = -1
        if config.flow.squeeze:
            z = SqueezeLayer(2).inverse(z)
        return z, logdet_kl
    else:
        return flow_model(x, reverse=reverse)


def init_model(config, train_data, flow_model):
    flow_model.eval()
    if config.flow.squeeze:
        train_data = SqueezeLayer(2).forward(train_data)

    if config.flow.model == 'resflow':
        flow_model.module(train_data, None, reverse=False)
    elif config.flow.model == 'wolf':
        flow_model.module.init(train_data, y=None, init_scale=1.0)
    else:
        raise NotImplementedError
    return flow_model


def create_flow_model(config):
    if config.flow.model == 'resflow':
        # Model
        if config.flow.squeeze:
            input_shape = (config.training.batch_size, config.data.num_channels *4, config.data.image_size // 2, config.data.image_size // 2)
        else:
            input_shape = (config.training.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
        flow_model = ResidualFlow(config,
            input_shape,
            n_blocks=list(map(int, config.flow.nblocks.split('-'))),
            intermediate_dim=config.flow.intermediate_dim,
            vnorms='ffff',
            actnorm=config.flow.actnorm,
            grad_in_forward=config.flow.grad_in_forward,
            activation_fn=config.flow.act_fn).to(config.device)
    elif config.flow.model == 'wolf':
        params = json.load(open(config.flow.model_config, 'r'))
        flow_model = WolfCore.from_params(params, config)
        flow_model.add_config(config)
    else:
        raise NotImplementedError

    flow_model = flow_model.to(config.device)
    flow_model = torch.nn.DataParallel(flow_model)

    return flow_model