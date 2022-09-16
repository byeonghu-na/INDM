import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 13000001
  training.snapshot_freq = 10000
  training.log_freq = 100
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = False
  training.importance_sampling = True
  training.unbounded_parametrization = False
  training.ddpm_score = True
  training.st = False
  training.k = 1.2
  training.truncation_time = 1e-5
  training.num_train_data = 50000
  training.reconstruction_loss = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15
  sampling.batch_size = 1024
  sampling.truncation_time = 1e-5

  sampling.temperature = 1.
  sampling.need_sample = True
  sampling.idx_rand = True
  sampling.pc_denoise = False
  sampling.pc_denoise_time = 0.
  sampling.more_step = False
  sampling.num_scales = 1000
  sampling.pc_ratio = 1.

  sampling.begin_snr = 0.16
  sampling.end_snr = 0.16
  sampling.snr_scheduling = 'none'

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 26
  evaluate.batch_size = 200
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = True
  evaluate.bpd_dataset = 'test'
  evaluate.num_test_data = 19962
  evaluate.residual = False
  evaluate.score_ema = True
  evaluate.flow_ema = False
  evaluate.num_nelbo = 3
  evaluate.rtol = 1e-5
  evaluate.atol = 1e-5

  evaluate.gap_diff = False
  evaluate.target_ckpt = -1
  evaluate.truncation_time = -1.

  evaluate.data_mean = False
  evaluate.skip_nll_wrong = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CELEBA'
  data.image_size = 64
  data.random_flip = True
  data.centered = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.auxiliary_resblock = True
  model.attention = True
  model.fourier_feature = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'AdamW'
  optim.weight_decay = 0.01
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = 1.
  optim.num_micro_batch = 1
  optim.reset = True
  optim.amsgrad = False

  # flow
  config.flow = flow = ml_collections.ConfigDict()
  flow.model = 'identity'
  flow.lr = 1e-3
  flow.ema_rate = 0.999
  flow.optim_reset = False
  flow.nblocks = '16-16'
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow'
  flow.squeeze = True
  flow.actnorm = False
  flow.grad_in_forward = False
  flow.act_fn = 'sin'

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  config.datadir = '.'
  config.checkpoint_meta_dir = '.'
  config.resume = False

  return config