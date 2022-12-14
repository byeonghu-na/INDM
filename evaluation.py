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

"""Utility functions for computing FID/Inception scores."""

import six
import logging
import os
import io
import torch
import numpy as np
import socket
import datasets
import sampling_lib

from cleanfid import fid as fid_calculator


import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(config, assetdir):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = assetdir + '/cifar10_stats.npz'
  elif config.data.dataset == 'IMAGENET32':
    filename = assetdir + '/imagenet32_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = assetdir + '/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_{mode}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = torch.cuda.device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }

def compute_fid_and_is(config, score_model, flow_model, sampling_fn, step, sample_dir, assetdir, num_data, eval=False,
                       inverse_scaler=None, this_sample_dir=None, scaler=None, data_mean=None):
  ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  ip.connect(("8.8.8.8", 80))
  ip = ip.getsockname()[0]

  inceptionv3 = config.data.image_size >= 256
  if config.data.dataset in ['CIFAR10', 'IMAGENET32']:
    samples_dir = tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npz"))
    # Use inceptionV3 for images with resolution higher than 256.

    for sample_name in samples_dir:
      sampling_idx = int(sample_name.split('/')[-1].split('_')[1].split('.')[0])
      # sampling_idx = 0
      _, samples = sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx,
                                            sample_dir, temperature=config.sampling.temperature, this_sample_dir=this_sample_dir,
                                            inverse_scaler=inverse_scaler, scaler=scaler, data_mean=data_mean)
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    samples_dir = tf.io.gfile.glob(os.path.join(this_sample_dir, "sample*.npz"))
    for sample_name in samples_dir:
      sampling_idx = int(sample_name.split('/')[-1].split('_')[1].split('.')[0])
      # sampling_idx = 0
      _, samples = sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx,
                                            sample_dir, temperature=config.sampling.temperature, this_sample_dir=this_sample_dir,
                                            inverse_scaler=inverse_scaler, scaler=scaler, data_mean=data_mean)
      latents = sampling_lib.get_latents(config, samples, inception_model, inceptionv3, step, sampling_idx, this_sample_dir)
      sampling_lib.save_statistics(config, latents, inceptionv3, step, sampling_idx, this_sample_dir)
    del inception_model
  compute_fid_and_is_(config, assetdir, config.data.image_size >= 256, step, [],
                                  sample_dir=this_sample_dir, num_data=num_data)
  tf.keras.backend.clear_session()
  torch.cuda.empty_cache()

def compute_fid_and_is_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='',
                       num_data=1000):
  if config.data.dataset in ['CIFAR10', 'IMAGENET32']:
    compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents,
                               num_data=num_data)
  elif config.data.dataset in ['FFHQ', 'LSUN', 'CelebAHQ', 'CELEBA', 'STL10', 'IMAGENET64', 'CIFAR100']:
    compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name=name, sample_dir=sample_dir, latents=latents,
                 num_data=num_data)
    if config.data.dataset == 'STL10':
      compute_is_stl10(config, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents)
  else:
    raise NotImplementedError


def compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='', num_data=1000):
  # if config.data.dataset == 'LSUN':
  #    fids = fid_ttur.calculate_fid_given_paths([sample_dir, assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_stats.npz'], './', low_profile=False)
  # elif config.data.dataset == 'FFHQ':

  # Mine
  fids = fid_calculator.compute_fid(config=config, mode='clean', fdir1=sample_dir, sigma_min=config.model.sigma_min,
                                    dataset_name=config.data.dataset, assetdir=assetdir, dataset=dataset,
                                    dequantization=True,
                                    num_data=num_data)

  # else:
  #    raise NotImplementedError
  print(fids)
  logging.info(f"{sample_dir}_ckpt-%d_{name} --- FID: {fids}" % (ckpt))

  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:
    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, fids=fids)
      f.write(io_buffer.getvalue())


def compute_is_stl10(config, inceptionv3, ckpt, name='/0', sample_dir='', latents=''):
  all_logits = []
  all_pools = []
  if latents == '':
    stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        print("stat : ", stat)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
  else:
    if not inceptionv3:
      all_logits.append(latents["logits"])
    all_pools.append(latents["pool_3"])
  if not inceptionv3:
    all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  all_use = False
  if all_use:
    inception_score = calculate_inception_score_CDSM(config, all_logits, inceptionv3)
  else:
    inception_score = calculate_inception_score_styleGAN(all_logits, inceptionv3, ckpt, name, sample_dir)

  logging.info(f'Inception score: {np.mean(inception_score)}')
  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:
    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score)
      f.write(io_buffer.getvalue())


def compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name='/0', sample_dir='', latents='',
                               num_data=None):
  # Compute inception scores, FIDs and KIDs.
  # Load all statistics that have been previously computed and saved for each host
  all_logits = []
  all_pools = []

  if latents == '':
    if config.sampling.pc_denoise:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, f"statistics_denoise_{config.sampling.pc_denoise_time}_*.npz"))
    elif config.sampling.more_step:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, f"statistics_more_step_*.npz"))
    else:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
    logging.info(f'sample_dir: {sample_dir}')
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        print("stat : ", stat)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
  else:
    if not inceptionv3:
      all_logits.append(latents["logits"])
    all_pools.append(latents["pool_3"])

  if num_data != None:
    num_samples = num_data
  else:
    num_samples = 50000

  if not inceptionv3:
    all_logits = np.concatenate(all_logits, axis=0)
  all_pools = np.concatenate(all_pools, axis=0)

  for k in range(len(all_logits) // num_samples):

    # Load pre-computed dataset statistics.
    print(f"assetdir: {assetdir}")
    data_stats = load_dataset_stats(config, assetdir)
    data_pools = data_stats["pool_3"]

    all_use = True
    if all_use:
      inception_score = calculate_inception_score_CDSM(all_logits[k * num_samples: (k + 1) * num_samples],
                                                       inceptionv3)
    else:
      inception_score = calculate_inception_score_styleGAN(all_logits[k * num_samples: (k + 1) * num_samples],
                                                           inceptionv3, ckpt, name, sample_dir)

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools[k * num_samples: (k + 1) * num_samples])
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools[k * num_samples: (k + 1) * num_samples])
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools
    name = name.split('/')[-1]

    logging.info(
      f"{sample_dir}_ckpt-%d_{name}_num_data-{num_data} --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
        ckpt, np.mean(inception_score), fid, kid))

  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:

    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())


def calculate_inception_score_styleGAN(all_logits, inceptionv3, ckpt, name, sample_dir):
  inception_scores = []
  for k in range(10):
    # indices = np.arange(k * 5000,(k+1) * 5000)

    if not inceptionv3:
      all_logit = all_logits[k * 5000: (k + 1) * 5000]

    print("all logits length : ", len(all_logit))
    assert len(all_logit) == 5000

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logit)
      inception_scores.append(inception_score)
    else:
      inception_score = -1

    logging.info(
      f"{sample_dir}_ckpt-%d_{name} --- inception_score: %.6e" % (
        ckpt, inception_score))

  return inception_scores

def calculate_inception_score_CDSM(all_logits, inceptionv3):
  print("all logits length : ", len(all_logits))
  # assert len(all_logits) == config.eval.num_samples

  # Compute FID/KID/IS on all samples together.
  if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
  else:
    inception_score = -1

  return inception_score

def get_bpd(config, eval_ds, scaler, nelbo_fn, nll_fn, score_model, flow_model=None, step=0, eval=False):
  with torch.no_grad():
    if config.flow.model != 'identity':
      flow_model.eval()
    if eval:
      num_data = config.eval.num_test_data
    else:
      num_data = 10000

    # num_data = 1000

    # nelbo
    nelbo_bpds_full = []
    nelbo_residual_bpds_full = []
    for _ in range(config.eval.num_nelbo):
      nelbo_bpds = []
      nelbo_residual_bpds = []
      bpd_iter = iter(eval_ds)
      for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
        eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)

        eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
        logdet = None
        eval_batch = scaler(eval_batch)
        nelbo_bpd, nelbo_residual_bpd = nelbo_fn(score_model, flow_model, eval_batch, logdet)
        nelbo_bpd = nelbo_bpd.detach().cpu().numpy().reshape(-1)
        nelbo_bpds.extend(nelbo_bpd)

        nelbo_residual_bpd = nelbo_residual_bpd.detach().cpu().numpy().reshape(-1)
        nelbo_residual_bpds.extend(nelbo_residual_bpd)
      nelbo_bpds_full.append(np.mean(nelbo_bpds))
      nelbo_residual_bpds_full.append(np.mean(nelbo_residual_bpds))
      logging.info("step: %d, num samples: %d, mean nelbo bpd: %.5e, std nelbo bpd: %.5e" % (
      step, len(nelbo_bpds), np.mean(nelbo_bpds), np.std(nelbo_bpds)))
      logging.info("step: %d, num samples: %d, mean nelbo_residual bpd: %.5e, std nelbo_residual bpd: %.5e" % (
      step, len(nelbo_residual_bpds), np.mean(nelbo_residual_bpds), np.std(nelbo_residual_bpds)))
    logging.info("step: %d, average nelbo bpd out of %d evaluations: %.5e" % (
      step, len(nelbo_bpds_full), np.mean(nelbo_bpds_full)))
    logging.info("step: %d, average nelbo bpd out of %d evaluations: %.5e" % (
      step, len(nelbo_residual_bpds_full), np.mean(nelbo_residual_bpds_full)))

    if not eval:
      num_data = num_data // 10

    # nll wrong with eps=1e-5
    if config.eval.skip_nll_wrong:
      pass
    else:
      for _ in range(1):
        if config.eval.truncation_time == -1.:
          eps_bpd = 1e-5
        else:
          eps_bpd = config.eval.truncation_time
        nll_bpds = []
        bpd_iter = iter(eval_ds)
        for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
          eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
          eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
          logdet = None
          eval_batch = scaler(eval_batch)
          nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=False, eps_bpd=eps_bpd)[0].detach().cpu().numpy().reshape(-1)
          nll_bpds.extend(nll_bpd)
          if eval:
            logging.info("step: %d, [NLL WRONG w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
              step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
        logging.info("step: %d, [NLL WRONG w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
        step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))

    # nll correct with eps=1e-5
    for _ in range(1):
      if config.eval.truncation_time == -1.:
        eps_bpd = 1e-5
      else:
        eps_bpd = config.eval.truncation_time
      nll_bpds = []
      bpd_iter = iter(eval_ds)
      for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
        eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
        eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
        logdet = None
        eval_batch = scaler(eval_batch)
        nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=True, eps_bpd=eps_bpd)[0].detach().cpu().numpy().reshape(-1)
        nll_bpds.extend(nll_bpd)
        if eval:
          logging.info("step: %d, [NLL CORRECT w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
            step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
      logging.info("step: %d, [NLL CORRECT w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
      step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))

    # nll correct with their eps if eps != 1e-5
    if config.training.truncation_time != 1e-5:
      for _ in range(1):
        nll_bpds = []
        bpd_iter = iter(eval_ds)
        for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
          eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
          eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
          logdet = None
          eval_batch = scaler(eval_batch)
          nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=True, eps_bpd=config.training.truncation_time)[0].detach().cpu().numpy().reshape(-1)
          nll_bpds.extend(nll_bpd)
          if eval:
            logging.info("step: %d, [NLL CORRECT w/ eps=eps] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
              step, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
        logging.info("step: %d, [NLL CORRECT w/ eps=eps] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
        step, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
    if config.flow.model != 'identity':
      flow_model.train()