# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import pickle
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from flow_models.flow_model import flow_forward
from flow_models.resflow.utils import update_lipschitz
import likelihood


def get_optimizer(config, params, lr=None, beta1=None, eps=None, weight_decay=None):
  """Returns a flax optimizer object based on `config`."""
  if lr is None: lr = config.optim.lr
  if beta1 is None: beta1 = config.optim.beta1
  if eps is None: eps = config.optim.eps
  if weight_decay is None: weight_decay = config.optim.weight_decay

  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=lr, betas=(beta1, 0.999), eps=eps, weight_decay=weight_decay, amsgrad=config.optim.amsgrad)
  elif config.optim.optimizer == 'AdamW':
    optimizer = optim.AdamW(params, lr=lr, betas=(beta1, 0.99), eps=eps, weight_decay=weight_decay, amsgrad=config.optim.amsgrad)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(config, sde, train, variance='scoreflow'):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if config.training.reduce_mean else torch.sum

  def loss_fn(model, batch, st=False, recon_loss=None, importance_sampling=None):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if recon_loss is None:
      recon_loss = config.training.reconstruction_loss
    if importance_sampling is None:
      importance_sampling = config.training.importance_sampling
    # if st:
    #   assert not recon_loss
    t_min = sde.get_t_min(config, st)
    t, Z = sde.get_diffusion_time(config, batch.shape[0], batch.device, t_min, importance_sampling=importance_sampling)

    score_fn = mutils.get_score_fn(config, sde, model, None, train=train, continuous=config.training.continuous)
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if importance_sampling:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      if config.training.likelihood_weighting:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
      else:
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    if recon_loss:
      eps_vec = torch.ones((batch.shape[0]), device=batch.device) * t_min
      mean, std = sde.marginal_prob(batch, eps_vec)
      z = torch.randn_like(batch)
      perturbed_data = mean + std[:, None, None, None] * z
      score = score_fn(perturbed_data, eps_vec)

      alpha, beta = sde.marginal_prob(torch.ones_like(batch), eps_vec)
      q_mean = perturbed_data / alpha + beta[:, None, None, None] ** 2 * score / alpha
      if variance == 'ddpm':
        q_std = beta
      elif variance == 'scoreflow':
        q_std = beta / torch.mean(alpha, axis=(1, 2, 3))

      n_dim = np.prod(batch.shape[1:])
      p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(std) + 1.)
      q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std)) + 0.5 / (q_std ** 2) * torch.square(batch - q_mean).sum(axis=(1, 2, 3))
      reconstruction_loss = q_recon - p_entropy
      if config.training.reduce_mean:
        reconstruction_loss = reconstruction_loss / n_dim
      losses = losses + reconstruction_loss

    return losses

  return loss_fn


def get_smld_loss_fn(config, vesde, train):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if config.training.reduce_mean else torch.sum

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(config, vpsde, train):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if config.training.reduce_mean else torch.sum

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(config, sde, train, optimize_fn=None, scaler=None):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if config.training.continuous:
    loss_fn = get_sde_loss_fn(config, sde, train)
  else:
    assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(config, sde, train)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(config, sde, train)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def calculate_logp(batch):
    Ts = torch.ones(batch.shape[0], device=config.device) * sde.T
    meanT, stdT = sde.marginal_prob(batch, Ts)
    z = torch.randn_like(batch)
    yT = meanT + stdT[:, None, None, None] * z
    log_p = sde.prior_logp(yT)
    return log_p

  def step_fn(state, flow_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      losses: The list of loss values of this state and batch.
    """
    model = state['model']
    optimizer = state['optimizer']

    optimizer.zero_grad()
    batch_size = batch.shape[0]
    num_micro_batch = config.optim.num_micro_batch
    losses_ = torch.zeros(batch_size)
    for k in range(num_micro_batch):
      losses = loss_fn(model, batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)])
      torch.mean(losses).backward(retain_graph=True)
      losses_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses.cpu().detach()
    optimize_fn(optimizer, model.parameters(), step=state['step'])
    state['step'] += 1
    state['ema'].update(model.parameters())

    return losses_, None, None, None, None

  def flow_step_fn_nll(state, flow_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      flow_state: A dictionary of training information, containing the flow model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      losses: The list of loss values of this state and batch.
    """
    model = state['model']
    flow_model = flow_state['model']
    optimizer = state['optimizer']
    flow_optimizer = flow_state['optimizer']

    batch_size = batch.shape[0]
    num_micro_batch = config.optim.num_micro_batch
    losses_ = torch.zeros(batch_size)
    losses_score_ = torch.zeros(batch_size)
    losses_flow_ = torch.zeros(batch_size)
    losses_logp_ = torch.zeros(batch_size)

    optimizer.zero_grad()
    flow_optimizer.zero_grad()

    if train:
      for k in range(num_micro_batch):
        mini_batch = batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)]
        transformed_mini_batch, losses_flow = flow_forward(config, flow_model, mini_batch, reverse=False)

        losses_score = loss_fn(model, transformed_mini_batch, st=config.training.st)
        losses_logp = calculate_logp(transformed_mini_batch)
        if config.training.reduce_mean:
          losses_flow = - losses_flow / np.prod(batch.shape[1:])
          losses_logp = - losses_logp / np.prod(batch.shape[1:])
        else:
          losses_flow = - losses_flow
          losses_logp = - losses_logp
        assert losses_score.shape == losses_flow.shape == losses_logp.shape == torch.Size([transformed_mini_batch.shape[0]])
        losses = losses_score + losses_flow + losses_logp
        torch.mean(losses).backward(retain_graph=True)
        # save losses
        losses_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses.cpu().detach()
        losses_score_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_score.cpu().detach()
        losses_flow_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_flow.cpu().detach()
        losses_logp_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_logp.cpu().detach()

      optimize_fn(optimizer, model.parameters(), step=state['step'])
      optimize_fn(flow_optimizer, flow_model.parameters(), step=flow_state['step'])

    update_lipschitz(flow_model)
    state['step'] += 1
    state['ema'].update(model.parameters())
    flow_state['step'] += 1
    flow_state['ema'].update(flow_model.parameters())

    return losses_, losses_score_, losses_flow_, losses_logp_

  def flow_step_fn_fid(state, flow_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      flow_state: A dictionary of training information, containing the flow model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      losses: The list of loss values of this state and batch.
    """
    model = state['model']
    flow_model = flow_state['model']
    optimizer = state['optimizer']
    flow_optimizer = flow_state['optimizer']

    batch_size = batch.shape[0]
    num_micro_batch = config.optim.num_micro_batch
    losses_ = torch.zeros(batch_size)
    losses_score_ = torch.zeros(batch_size)
    losses_flow_ = torch.zeros(batch_size)
    losses_logp_ = torch.zeros(batch_size)

    optimizer.zero_grad()
    flow_optimizer.zero_grad()

    if train:
      # flow training (all losses)
      for k in range(num_micro_batch):
        mini_batch = batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)]
        transformed_mini_batch, losses_flow = flow_forward(config, flow_model, mini_batch, reverse=False)

        losses_score = loss_fn(model, transformed_mini_batch, importance_sampling=True)
        losses_logp = calculate_logp(transformed_mini_batch)
        if config.training.reduce_mean:
          losses_flow = - losses_flow / np.prod(batch.shape[1:])
          losses_logp = - losses_logp / np.prod(batch.shape[1:])
        else:
          losses_flow = - losses_flow
          losses_logp = - losses_logp
        assert losses_score.shape == losses_flow.shape == losses_logp.shape == torch.Size([transformed_mini_batch.shape[0]])
        losses = losses_score + losses_flow + losses_logp
        torch.mean(losses).backward(retain_graph=True)
        # save losses
        losses_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses.cpu().detach()
        losses_flow_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_flow.cpu().detach()
        losses_logp_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_logp.cpu().detach()
      optimize_fn(flow_optimizer, flow_model.parameters(), step=flow_state['step'])
      update_lipschitz(flow_model)
      flow_state['ema'].update(flow_model.parameters())

      # diffusion training with st
      if not config.training.st:
        optimizer.zero_grad()
      for k in range(num_micro_batch):
        mini_batch = batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)]
        if not config.training.st:
          with torch.no_grad():
            transformed_mini_batch, _ = flow_forward(config, flow_model, mini_batch, log_det=None, reverse=False)
        transformed_mini_batch = transformed_mini_batch.detach()
        losses_add_score = loss_fn(model, transformed_mini_batch, st=config.training.st, recon_loss=False)

        if config.training.st:
          const_adj = (losses_add_score.mean() / losses_score.mean()).detach()
          for p in model.parameters():
            if p.grad is None:
              continue
            else:
              p.grad = const_adj * p.grad

        torch.mean(losses_add_score).backward(retain_graph=True)
        # save losses
        losses_score_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses_add_score.cpu().detach()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['ema'].update(model.parameters())

      state['step'] += 1
      flow_state['step'] += 1

    return losses_, losses_score_, losses_flow_, losses_logp_

  if config.flow.model == 'identity':
    print('Train only the score network.')
    return step_fn
  elif config.flow.model != 'identity':
    print('Train flow network with NLL.')
    if not config.training.likelihood_weighting:
      print('Train score network with FID-favorable setting (weighting function = variance weighting).')
      return flow_step_fn_fid
    else:
      print('Train score network with NLL-favorable setting (weighting function = likelihood weighting).')
      return flow_step_fn_nll
  else:
    raise NotImplementedError


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      #x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    #x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn
