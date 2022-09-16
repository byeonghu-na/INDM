import torch
import tensorflow as tf
import os
import logging
import numpy as np

from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
import likelihood, sampling
import flowpp_models
from flow_models.flow_model import create_flow_model

def restore_checkpoint(config, ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    logging.info(ckpt_dir + ' loaded ...')
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if config.training.sde != 'vesde':
      state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    if config.model.name == 'vdm':
      state['gamma_optimizer'].load_state_dict(loaded_state['gamma_optimizer'])
      state['gamma'] = loaded_state['gamma_minmax']
      state['noise_optimizer'].load_state_dict(loaded_state['noise_optimizer'])
      state['noise_schedule'].load_state_dict(loaded_state['schedule_model'])
    return state


def save_checkpoint(config, ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  if config.model.name == 'vdm':
    saved_state['gamma'] = state['gamma']
    saved_state['gamma_optimizer'] = state['gamma_optimizer'].state_dict()
    saved_state['noise_schedule'] = state['schedule_model'].state_dict()
    saved_state['noise_optimizer'] = state['schedule_optimizer'].state_dict()
  torch.save(saved_state, ckpt_dir)

def create_name(prefix, name, ext):
  try:
    name = f'{prefix}_{int(name)}.{ext}'
  except:
    if len(name.split('.')) == 1:
      name = f'{prefix}_{name}.{ext}'
    else:
      name = name.split('/')[-1]
      name = f'{prefix}_{name.split(".")[0]}.{ext}'
  return name

def load_model(config, workdir, print=True):
  # Initialize model.
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  if config.model.name == 'vdm':
    from models import vdm
    gamma_minmax = torch.randn(2, device=config.device).requires_grad_(True)
    schedule_model = torch.nn.DataParallel(vdm.Noise_Schedule().to(config.device))
    gamma_optimizer = losses.get_optimizer(config, [gamma_minmax])
    schedule_optimizer = losses.get_optimizer(config, schedule_model.parameters())
    state = dict(optimizer=optimizer, model=score_model,
                 gamma_optimizer=gamma_optimizer, gamma=gamma_minmax,
                 noise_optimizer=schedule_optimizer, noise_schedule=schedule_model, ema=ema, step=0)
  else:
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  if print:
    # print(score_model)
    model_parameters = filter(lambda p: p.requires_grad, score_model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    total_num_params = sum([np.prod(p.size()) for p in score_model.parameters()])
    logging.info(f"model parameters: {model_params}")
    logging.info(f"total number of parameters: {total_num_params}")

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  if config.eval.target_ckpt == -1:
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  else:
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints", f"checkpoint_{config.eval.target_ckpt}.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(config, checkpoint_meta_dir, state, config.device)

  if config.optim.reset:
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state['optimizer'] = optimizer

  # if we add flow, reset ema
  if config.flow.model != 'identity':
    flow_checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "flow_checkpoint.pth")
    if not tf.io.gfile.exists(flow_checkpoint_meta_dir):
      logging.info(f"No flow checkpoints, so reset score ema!!")
      ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
      state['ema'] = ema
    else:
      logging.info(f"There exists flow checkpoints, so keep score ema!!")

  return state, score_model, ema, checkpoint_dir, checkpoint_meta_dir

def load_flow_model(config, workdir, print=True):
  # Initialize model.
  flow_model = create_flow_model(config)
  flow_optimizer = losses.get_optimizer(config, flow_model.parameters(), lr=config.flow.lr)
  flow_ema = ExponentialMovingAverage(flow_model.parameters(), decay=config.flow.ema_rate)
  flow_state = dict(optimizer=flow_optimizer, model=flow_model, ema=flow_ema, step=0)

  if print:
    # print(score_model)
    model_parameters = filter(lambda p: p.requires_grad, flow_model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    total_num_params = sum([np.prod(p.size()) for p in flow_model.parameters()])
    logging.info(f"model parameters: {model_params}")
    logging.info(f"total number of parameters: {total_num_params}")

  # Create checkpoints directory
  flow_checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  if config.eval.target_ckpt == -1:
    flow_checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "flow_checkpoint.pth")
  else:
    flow_checkpoint_meta_dir = os.path.join(workdir, "checkpoints", f"flow_checkpoint_{config.eval.target_ckpt}.pth")
  tf.io.gfile.makedirs(flow_checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(flow_checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  flow_state = restore_checkpoint(config, flow_checkpoint_meta_dir, flow_state, config.device)

  if config.flow.optim_reset:
    flow_optimizer = losses.get_optimizer(config, flow_model.parameters())
    flow_state['optimizer'] = flow_optimizer

  return flow_state, flow_model, flow_ema, flow_checkpoint_dir, flow_checkpoint_meta_dir

def get_loss_fns(config, sde, inverse_scaler, train=True, scaler=None):
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(config, sde, train=train, optimize_fn=optimize_fn, scaler=scaler)
  nll_fn = likelihood.get_likelihood_fn(config, sde, inverse_scaler, rtol=config.eval.rtol, atol=config.eval.atol)
  nelbo_fn = likelihood.get_elbo_fn(config, sde, inverse_scaler=inverse_scaler)
  sampling_shape = (config.sampling.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, config.sampling.truncation_time)
  return train_step_fn, nll_fn, nelbo_fn, sampling_fn