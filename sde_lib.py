"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape, data_mean=None):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def get_diffusion_time(self, config):
    pass

  def discretize(self, x, t, next_t=None):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, next_t=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if next_t is None:
          f, G = discretize_fn(x, t, next_t)
        else:
          if next_t[0].item() > 0:
            f, G = discretize_fn(x, t, next_t)
          else:
            f = torch.zeros(x.shape, device=x.device)
            _, G = sde_fn(x, t)
            G = G * torch.sqrt(t - next_t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, truncation_time=1e-5, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.eps = truncation_time
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape, data_mean=None):
    # return torch.randn(*shape)
    if data_mean is None:
      data_mean = 0.
    return torch.randn(*shape) + data_mean

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t, next_t=None):
    """DDPM discretization."""
    if next_t == None:
      timestep = (t * (self.N - 1) / self.T).long()
      beta = self.discrete_betas.to(x.device)[timestep]
      alpha = self.alphas.to(x.device)[timestep]
      sqrt_beta = torch.sqrt(beta)
      f = torch.sqrt(alpha)[:, None, None, None] * x - x
      G = sqrt_beta
    else:
      G = torch.sqrt((t - next_t) * (self.beta_0 + (self.beta_1 - self.beta_0) * t))
      f = torch.sqrt(1. - G ** 2)[:, None, None, None] * x - x
      #f = - 0.5 * (G ** 2)[:, None, None, None] * x
    return f, G

  def integral_beta(self, t):
    return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0

  def antiderivative(self, t, stabilizing_constant=0.):
    if isinstance(t, float) or isinstance(t, int):
      t = torch.tensor(t).float()
    return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

  def normalizing_constant(self, t_min):
    return self.antiderivative(self.T) - self.antiderivative(t_min)

  def get_diffusion_time(self, config, batch_size, batch_device, t_min, importance_sampling=None):
    if importance_sampling is None:
      importance_sampling = config.training.importance_sampling
    if importance_sampling:
      Z = self.normalizing_constant(t_min)
      u = torch.rand(batch_size, device=batch_device)
      return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) * \
                torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
    else:
      return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

  def get_t_min(self, config, st=False):
    if st:
      if config.training.k == 1.0:
        return self.eps ** (1. - np.random.rand())
      else:
        return self.eps / (1. - np.random.rand() * (1 - self.eps ** (config.training.k - 1))) ** (1. / (config.training.k - 1))
    else:
      return self.eps

class subVPSDE(SDE):
  def __init__(self, truncation_time=1e-5, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape, data_mean=None):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, truncation_time=1e-5, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.eps = truncation_time
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape, data_mean=None):
    # return torch.randn(*shape) * self.sigma_max
    if data_mean is None:
      data_mean = 0.
    return torch.randn(*shape) * self.sigma_max + data_mean

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  #def discretize(self, x, t, next_t):
  #  """SMLD(NCSN) discretization."""
  #  timestep = (t * (self.N - 1) / self.T).long()
  #  sigma = self.discrete_sigmas.to(t.device)[timestep]
  #  adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
  #                               self.discrete_sigmas[timestep - 1].to(t.device))
  #  f = torch.zeros_like(x)
  #  G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
  #  return f, G

  def discretize(self, x, t, next_t):
    if next_t is None:
      timestep = (t * (self.N - 1) / self.T).long()
      sigma = self.discrete_sigmas.to(t.device)[timestep]
      adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                   self.discrete_sigmas[timestep - 1].to(t.device))
      f = torch.zeros_like(x)
      G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    else:
      _, std_t = self.marginal_prob(x, t)
      _, std_next_t = self.marginal_prob(x, next_t)
      f = torch.zeros_like(x)
      G = torch.sqrt(std_t ** 2 - std_next_t ** 2)
    return f, G

  def antiderivative(self, t):
    if isinstance(t, float) or isinstance(t, int):
      t = torch.tensor(t).float()
    return 2. * torch.log(self.sigma_min * (self.sigma_max / self.sigma_min) ** t)

  def normalizing_constant(self, t_min):
    return self.antiderivative(self.T) - self.antiderivative(t_min)

  def get_diffusion_time(self, config, batch_size, batch_device, t_min, importance_sampling=None):
    if importance_sampling is None:
      importance_sampling = config.training.importance_sampling
    if importance_sampling:
      Z = self.normalizing_constant(t_min)
      u = torch.rand(batch_size, device=batch_device)
      return t_min + ((Z * u) / (2. * (np.log(self.sigma_max) - np.log(self.sigma_min)))), Z.detach()
    else:
      return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

  def get_t_min(self, config, st=False):
    if st:
      if config.training.k == 1.0:
        return self.eps ** (1. - np.random.rand())
      else:
        return self.eps / (1. - np.random.rand() * (1 - self.eps ** (config.training.k - 1))) ** (1. / (config.training.k - 1))
    else:
      return self.eps


class GeometricVPSDE(VPSDE):
  def __init__(self, truncation_time=1e-5, beta_min=0.1, beta_max=20, N=1000, sigma2_min=3*1e-5, sigma2_max=0.999):
    """Construct a Geometric Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(truncation_time, beta_min, beta_max, N)
    self.sigma2_0 = sigma2_min
    self.sigma2_min = sigma2_min
    self.sigma2_max = sigma2_max

    self.beta_0 = (self.sigma2_min / (1. - self.sigma2_min)) * np.log(self.sigma2_max / self.sigma2_min)
    self.beta_1 = (self.sigma2_max / (1. - self.sigma2_max)) * np.log(self.sigma2_max / self.sigma2_min)

    self.eps = truncation_time
    self.N = N

    # discrete_betas  /  alphas ????
    t = torch.linspace(0, 1, N)
    sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
    log_term = np.log(self.sigma2_max / self.sigma2_min)
    self.discrete_betas = sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
    log_term = np.log(self.sigma2_max / self.sigma2_min)
    beta_t = sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    # log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    mean = torch.sqrt(1.0 + self.sigma2_min * (1.0 - (self.sigma2_max / self.sigma2_min) ** t[:, None, None, None]) / (1.0 - self.sigma2_0)) * x
    # std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    std = torch.sqrt(self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0)

    return mean, std

  def prior_sampling(self, shape, data_mean=None):
    # return torch.randn(*shape)
    if data_mean is None:
      data_mean = 0.
    return torch.randn(*shape) + data_mean

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t, next_t=None):
    """DDPM discretization."""
    if next_t == None:
      raise NotImplementedError()
      timestep = (t * (self.N - 1) / self.T).long()
      beta = self.discrete_betas.to(x.device)[timestep]
      alpha = self.alphas.to(x.device)[timestep]
      sqrt_beta = torch.sqrt(beta)
      f = torch.sqrt(alpha)[:, None, None, None] * x - x
      G = sqrt_beta
    else:
      # G = torch.sqrt((t - next_t) * (self.beta_0 + (self.beta_1 - self.beta_0) * t))
      sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
      log_term = np.log(self.sigma2_max / self.sigma2_min)
      beta_t = sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)
      G = torch.sqrt((t - next_t) * beta_t)
      f = torch.sqrt(1. - G ** 2)[:, None, None, None] * x - x
    return f, G

  def integral_beta(self, t):
    return torch.log((1. - self.sigma2_min) / (1. - self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)))

  def antiderivative(self, t, stabilizing_constant=0.):
    if isinstance(t, float) or isinstance(t, int):
      t = torch.tensor(t).float()
    return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

  def normalizing_constant(self, t_min):
    return self.antiderivative(self.T) - self.antiderivative(t_min)

  def get_diffusion_time(self, config, batch_size, batch_device, t_min, importance_sampling=None):
    # if importance_sampling is None:
    #   importance_sampling = config.training.importance_sampling
    # if importance_sampling:
    #   Z = self.normalizing_constant(t_min)
    #   u = torch.rand(batch_size, device=batch_device)
    #   exp = torch.exp(Z * u + self.antiderivative(t_min))
    #   rhs = torch.log(self.sigma2_min + exp) - torch.log(1. + exp) - torch.log(torch.tensor(self.sigma2_min, device=batch_device))
    #   return rhs / (torch.log(torch.tensor(self.sigma2_max / self.sigma2_min, device=batch_device))), Z.detach()
    # else:
    return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

  def get_t_min(self, config, st=False):
    if st:
      if config.training.k == 1.0:
        return self.eps ** (1. - np.random.rand())
      else:
        return self.eps / (1. - np.random.rand() * (1 - self.eps ** (config.training.k - 1))) ** (1. / (config.training.k - 1))
    else:
      return self.eps


def get_sde(config):
  if config.training.sde.lower() == 'vpsde':
    sde = VPSDE(truncation_time=config.training.truncation_time, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = subVPSDE(truncation_time=config.training.truncation_time, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = VESDE(truncation_time=config.training.truncation_time, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'gvpsde':
    sde = GeometricVPSDE(truncation_time=config.training.truncation_time, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  return sde