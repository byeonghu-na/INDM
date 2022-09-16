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

import logging
import os
import io
import torch
import numpy as np
import gc
import evaluation
import utils
import tensorflow as tf
from torchvision.utils import make_grid, save_image
from flow_models.flow_model import flow_forward


def get_samples(config, score_model, flow_model, sampling_fn, step, r, sample_dir, temperature=1., inverse_scaler=None, this_sample_dir=None, scaler=None, data_mean=None):
    logging.info("sampling -- ckpt step: %d, round: %d" % (step, r))
    tf.io.gfile.makedirs(sample_dir)
    tf.io.gfile.makedirs(this_sample_dir)
    if not os.path.exists(os.path.join(sample_dir, f'samples_{r}_before_flow.npz')):
        samples_before_flow, samples_after_flow, n = sampling_fn(score_model, flow_model, temperature, data_mean, sample_dir=sample_dir, r=r)
        logging.info(f'nfe: {n}')

        # save npz file of 'before_flow' samples
        samples = (samples_before_flow.permute(0, 2, 3, 1).cpu().numpy() * 255.)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        assert samples.shape == (samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{r}_before_flow.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
        # save npz file of 'after_flow' samples
        samples = np.clip(samples_after_flow.permute(0, 2, 3, 1).cpu().numpy() * 255., 0., 255.).astype(np.uint8)
        assert samples.shape == (samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        filename = f"samples_{r}.npz"
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, filename), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
    else:
        samples_before_flow = torch.tensor(np.load(os.path.join(sample_dir, f"samples_{r}_before_flow.npz"))['samples']).permute(0, 3, 1, 2) / 255.
        if config.sampling.pc_denoise:
            if not os.path.exists(os.path.join(this_sample_dir, f'samples_{r}_denoise_{config.sampling.pc_denoise_time}.npz')):
                if not os.path.exists(os.path.join(sample_dir, f'samples_{r}_before_flow_denoise_{config.sampling.pc_denoise_time}.npz')):
                    logging.info(f'denoise for pc with round {r} and final time {config.sampling.pc_denoise_time}')
                    if config.training.sde == 'vesde':
                        samples_before_flow = torch.tensor(np.load(os.path.join(sample_dir, f"samples_{r}_before_flow_for_search.npz"))['samples']).permute(0, 3, 1, 2) / 255.
                    else:
                        samples_before_flow = torch.tensor(np.load(os.path.join(sample_dir, f"samples_{r}_before_flow.npz"))['samples']).permute(0, 3, 1, 2) / 255.
                    samples_before_flow, samples_after_flow, n = sampling_fn(score_model, flow_model, temperature, data_mean, final_time=config.sampling.pc_denoise_time, before_data=scaler(samples_before_flow))
                    # save npz file of 'before_flow' samples
                    samples = (samples_before_flow.permute(0, 2, 3, 1).cpu().numpy() * 255.)
                    samples = samples.reshape(
                        (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                    assert samples.shape == (
                    samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
                    # Write samples to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{r}_before_flow_denoise_{config.sampling.pc_denoise_time}.npz"), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, samples=samples)
                        fout.write(io_buffer.getvalue())
                else:
                    samples_before_flow = torch.tensor(np.load(os.path.join(sample_dir, f"samples_{r}_before_flow_denoise_{config.sampling.pc_denoise_time}.npz"))['samples']).permute(0, 3, 1, 2) / 255.
                    with torch.no_grad():
                        if config.flow.model != 'identity':
                            assert samples_before_flow.shape[0] % 16 == 0
                            samples_after_flow = torch.zeros_like(samples_before_flow, device='cuda')
                            for k in range(samples_before_flow.shape[0] // 16):
                                samples_after_flow[16 * k:16 * (k + 1)] = \
                                    flow_forward(config, flow_model,
                                                 scaler(samples_before_flow[16 * k:16 * (k + 1)]).to(
                                                     'cuda') * temperature,
                                                 log_det=None, reverse=True)[0]
                        samples_after_flow = inverse_scaler(samples_after_flow)
                samples = np.clip(samples_after_flow.permute(0, 2, 3, 1).cpu().numpy() * 255., 0., 255.).astype(
                    np.uint8)
                assert samples.shape == (
                    samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}_denoise_{config.sampling.pc_denoise_time}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())
            else:
                print(f'load denoise sample with {config.sampling.pc_denoise_time}')
                samples_after_flow = torch.tensor(np.load(os.path.join(this_sample_dir, f"samples_{r}_denoise_{config.sampling.pc_denoise_time}.npz"))['samples']).permute(
                    0, 3, 1, 2) / 255.

        elif config.sampling.more_step:
            if not os.path.exists(os.path.join(this_sample_dir, f'samples_{r}_more_step.npz')):
                logging.info(f'more step with round {r}')
                samples_before_flow, samples_after_flow, n = sampling_fn(score_model, flow_model, temperature, data_mean, before_data=scaler(samples_before_flow))
                # save npz file of 'before_flow' samples
                samples = (samples_before_flow.permute(0, 2, 3, 1).cpu().numpy() * 255.)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                assert samples.shape == (
                samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{r}_before_flow_more_step.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                samples = np.clip(samples_after_flow.permute(0, 2, 3, 1).cpu().numpy() * 255., 0., 255.).astype(
                    np.uint8)
                assert samples.shape == (
                    samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}_more_step.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())
            else:
                print(f'load more step sample')
                samples_after_flow = torch.tensor(np.load(os.path.join(this_sample_dir, f"samples_{r}_more_step.npz"))['samples']).permute(
                    0, 3, 1, 2) / 255.

        else:
            if not os.path.exists(os.path.join(this_sample_dir, f'samples_{r}.npz')):
                with torch.no_grad():
                    if config.flow.model != 'identity':
                        assert samples_before_flow.shape[0] % 16 == 0
                        samples_after_flow = torch.zeros_like(samples_before_flow, device='cuda')
                        for k in range(samples_before_flow.shape[0] // 16):
                            samples_after_flow[16 * k:16 * (k + 1)] = \
                            flow_forward(config, flow_model, scaler(samples_before_flow[16 * k:16 * (k + 1)]).to('cuda') * temperature,
                                         log_det=None, reverse=True)[0]
                    samples_after_flow = inverse_scaler(samples_after_flow)
                samples = np.clip(samples_after_flow.permute(0, 2, 3, 1).cpu().numpy() * 255., 0., 255.).astype(np.uint8)
                assert samples.shape == (
                    samples.shape[0], config.data.image_size, config.data.image_size, config.data.num_channels)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                nrow = int(np.sqrt(samples.shape[0]))
                image_grid = make_grid(torch.tensor(samples).permute(0, 3, 1, 2) / 255., nrow, padding=2)
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"sample_{r}.png"), "wb") as fout:
                    save_image(image_grid, fout)
            else:
                samples_after_flow = torch.tensor(np.load(os.path.join(this_sample_dir, f"samples_{r}.npz"))['samples']).permute(
                    0, 3, 1, 2) / 255.

    return samples_before_flow, samples_after_flow

def get_latents(config, samples, inception_model, inceptionv3, step, r, sample_dir, small_batch=128):
    latents = {}
    num = (samples.shape[0] - 1) // small_batch + 1
    if config.sampling.pc_denoise:
        name = utils.create_name(f'statistics_denoise_{config.sampling.pc_denoise_time}', r, 'npz')
    elif config.sampling.more_step:
        name = utils.create_name(f'statistics_more_step', r, 'npz')
    else:
        name = utils.create_name('statistics', r, 'npz')
    samples = (samples.permute(0,2,3,1) * 255.).cpu().detach().numpy().astype(np.uint8)
    # samples = torch.tensor(samples, device=inception_model.device)
    if not os.path.exists(os.path.join(sample_dir, name)):
        for k in range(num):
            # Force garbage collection before calling TensorFlow code for Inception network
            gc.collect()
            latents_temp = evaluation.run_inception_distributed(samples[small_batch * k:small_batch * (k + 1)],
                                                                inception_model,
                                                                inceptionv3=inceptionv3)
            if k == 0:
                latents['pool_3'] = latents_temp['pool_3']
                if not inceptionv3:
                    latents['logits'] = latents_temp['logits']
            else:
                latents['pool_3'] = tf.concat([latents['pool_3'], latents_temp['pool_3']], 0)
                if not inceptionv3:
                    latents['logits'] = tf.concat([latents['logits'], latents_temp['logits']], 0)
            # Force garbage collection again before returning to JAX code
            gc.collect()
    else:
        latents = ''
    return latents

def save_statistics(config, latents, inceptionv3, step, r, sample_dir):
    if config.sampling.pc_denoise:
        name = utils.create_name(f'statistics_denoise_{config.sampling.pc_denoise_time}', r, 'npz')
    elif config.sampling.more_step:
        name = utils.create_name(f'statistics_more_step', r, 'npz')
    else:
        name = utils.create_name('statistics', r, 'npz')
    if not os.path.exists(os.path.join(sample_dir, name)):
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(sample_dir, name), "wb") as fout:
            io_buffer = io.BytesIO()
            if not inceptionv3:
                np.savez_compressed(
                    io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            else:
                np.savez_compressed(
                    io_buffer, pool_3=latents["pool_3"])
            fout.write(io_buffer.getvalue())