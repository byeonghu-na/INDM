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

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.likelihood_weighting = False
  training.importance_sampling = False

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  # flow
  flow = config.flow
  flow.model = 'wolf'
  flow.lr = 1e-3
  flow.ema_rate = 0.999
  flow.optim_reset = False
  flow.nblocks = '16-16'
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow'

  flow.model_config = 'flow_models/wolf/wolf_configs/cifar10/glow/resflow-gaussian-uni.json'
  flow.rank = 1
  flow.local_rank = 0
  flow.batch_size = 512
  flow.eval_batch_size = 4
  flow.batch_steps = 1
  flow.init_batch_size = 1024
  flow.epochs = 500
  flow.valid_epochs = 1
  flow.seed = 65537
  flow.train_k = 1
  flow.log_interval = 10
  # flow.lr = 0.001
  flow.warmup_steps = 500
  flow.lr_decay = 0.999997
  flow.beta1 = 0.9
  flow.beta2 = 0.999
  flow.eps = 1e-8
  flow.weight_decay = 0
  flow.amsgrad = True
  flow.grad_clip = 0
  flow.dataset = 'cifar10'
  flow.category = None
  flow.image_size = 32
  flow.workers = 4
  flow.n_bits = 8
  flow.recover = -1

  return config
