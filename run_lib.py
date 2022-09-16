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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """
import os
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, vdm
import sampling_lib
import socket
ip = socket.gethostbyname(socket.gethostname())
import datasets
import evaluation
import sde_lib
from absl import flags
import torch
import utils
import losses
import numpy as np
from flow_models.flow_model import flow_forward

FLAGS = flags.FLAGS


def train(config, workdir, assetdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)

  # Initialize model.
  state, score_model, ema, checkpoint_dir, checkpoint_meta_dir = utils.load_model(config, workdir)
  logging.info(f'score model step: {int(state["step"])}')
  initial_step = int(state['step'])

  # Initialze flow model
  if config.flow.model == 'identity':
    flow_state, flow_model = None, None
  else:
    flow_state, flow_model, flow_ema, flow_checkpoint_dir, flow_checkpoint_meta_dir = utils.load_flow_model(config, workdir)

  # Build data iterators
  logging.info(f'loading {config.data.dataset}...')
  train_ds, eval_ds = datasets.get_dataset(config)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde = sde_lib.get_sde(config)

  # Build one-step loss functions
  train_step_fn, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler, scaler=scaler)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, config.training.n_iters + 1):

    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch, train_iter = datasets.get_batch(config, train_iter, train_ds)
    batch = (255. * batch + torch.rand_like(batch)) / 256.
    batch = scaler(batch)

    # Execute one training step
    losses, losses_score, losses_flow, losses_logp = train_step_fn(state, flow_state, batch)
    if step % config.training.log_freq == 0:
      if config.flow.model == 'identity':
        logging.info("step: %d, training loss mean: %.5e, training loss std: %.5e" % (step, torch.mean(losses).item(), torch.std(losses).item()))
      else:
        logging.info("step: %d, loss mean: %.5e, score loss mean: %.5e, flow loss mean: %.5e, logp mean: %.5e"
                     % (step, torch.mean(losses).item(), torch.mean(losses_score).item(),
                        torch.mean(losses_flow).item(), torch.mean(losses_logp).item()))
        logging.info("step: %d, loss std: %.5e, score loss std: %.5e, flow loss std: %.5e, logp std: %.5e"
                    % (step, torch.std(losses).item(), torch.std(losses_score).item(),
                       torch.std(losses_flow).item(), torch.std(losses_logp).item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step != initial_step and step % config.training.snapshot_freq_for_preemption == 0:
      utils.save_checkpoint(config, checkpoint_meta_dir, state)
      if config.flow.model != 'identity':
        utils.save_checkpoint(config, flow_checkpoint_meta_dir, flow_state)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step != initial_step and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      utils.save_checkpoint(config, os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
      if config.flow.model != 'identity':
        utils.save_checkpoint(config, os.path.join(checkpoint_dir, f'flow_checkpoint_{save_step}.pth'), flow_state)

    if step != 0 and step != initial_step and step % config.training.snapshot_freq_for_preemption == 0:
      if config.eval.enable_bpd:
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        evaluation.get_bpd(config, eval_ds, scaler, nelbo_fn, nll_fn, score_model, flow_model=flow_model, step=step)
        ema.restore(score_model.parameters())
        torch.cuda.empty_cache()

    if step != 0 and step != initial_step and step % config.training.snapshot_freq_for_preemption == 0 or step == config.training.n_iters:
      this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
      # Generate and save samples
      if config.training.snapshot_sampling:
        logging.info('sampling start ...')
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        if config.flow.model != 'identity':
          flow_model.eval()
        for sampling_idx in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
          sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx, this_sample_dir, inverse_scaler=inverse_scaler, this_sample_dir=this_sample_dir)
        torch.cuda.empty_cache()
        logging.info('sampling end ... computing FID ...')
        evaluation.compute_fid_and_is(config, score_model, flow_model, sampling_fn, step, this_sample_dir, assetdir, config.eval.num_samples, this_sample_dir=this_sample_dir)
        ema.restore(score_model.parameters())
        if config.flow.model != 'identity':
          flow_model.train()

def evaluate(config,
             workdir,
             assetdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model.
  state, score_model, ema, checkpoint_dir, checkpoint_meta_dir = utils.load_model(config, workdir)
  logging.info(f'score model step: {int(state["step"])}')

  # Initialze flow model
  if config.flow.model == 'identity':
    flow_state, flow_model = None, None
  else:
    flow_state, flow_model, flow_ema, flow_checkpoint_dir, flow_checkpoint_meta_dir = utils.load_flow_model(config, workdir)

  # Setup SDEs
  sde = sde_lib.get_sde(config)

  # Build one-step loss functions
  _, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler, scaler=scaler)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds, eval_ds = datasets.get_dataset(config)

  if config.eval.enable_bpd:
    torch.cuda.empty_cache()
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    evaluation.get_bpd(config, eval_ds, scaler, nelbo_fn, nll_fn, score_model, flow_model=flow_model, step=int(state['step']), eval=True)
    ema.restore(score_model.parameters())
    torch.cuda.empty_cache()

  if config.eval.enable_sampling:
    if config.eval.data_mean:
      data_mean = 0.
      train_iter = iter(train_ds)
      if config.flow.model != 'identity':
        flow_model.eval()
      ema.copy_to(score_model.parameters())
      for batch_id in range((config.training.num_train_data - 1) // config.training.batch_size + 1):
        train_batch, _ = datasets.get_batch(config, train_iter, train_ds)
        with torch.no_grad():
          train_batch = (255. * train_batch + torch.rand_like(train_batch)) / 256.
          train_batch = scaler(train_batch)
          transformed_train_batch, _ = flow_forward(config, flow_model, train_batch, log_det=None, reverse=False)
          if config.training.sde != 'vesde':
            transformed_train_batch, _ = sde.marginal_prob(transformed_train_batch, torch.ones(transformed_train_batch.shape[0], device=transformed_train_batch.device))
        data_mean += transformed_train_batch.cpu().sum(0)
      data_mean = data_mean / config.training.num_train_data
      ema.restore(score_model.parameters())
      torch.cuda.empty_cache()
      if config.flow.model != 'identity':
        flow_model.train()
    else:
      data_mean = None

    if config.eval.target_ckpt == -1:
      sample_dir = os.path.join(workdir, "eval")
    else:
      sample_dir = os.path.join(workdir, "eval", f"ckpt_{config.eval.target_ckpt}")
    if config.sampling.temperature != 1.:
      this_sample_dir = os.path.join(sample_dir, f"temp_{config.sampling.temperature}")
    else:
      this_sample_dir = sample_dir
    step = int(state['step'])
    logging.info('sampling start ...')
    torch.cuda.empty_cache()
    if config.flow.model != 'identity':
      flow_model.eval()
    ema.copy_to(score_model.parameters())
    if config.sampling.need_sample:
      for sampling_idx in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
        if config.sampling.idx_rand:
          sampling_idx_rand = np.random.randint(0, 10000000)
        else:
          sampling_idx_rand = sampling_idx
        sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx_rand, sample_dir,
                                 temperature=config.sampling.temperature, inverse_scaler=inverse_scaler, this_sample_dir=this_sample_dir, scaler=scaler, data_mean=data_mean)
    # ema.restore(score_model.parameters())
    torch.cuda.empty_cache()
    logging.info('sampling end ... computing FID ...')
    evaluation.compute_fid_and_is(config, score_model, flow_model, sampling_fn, step, sample_dir, assetdir, config.eval.num_samples,
                                  inverse_scaler=inverse_scaler, eval=True, this_sample_dir=this_sample_dir, scaler=scaler, data_mean=data_mean)