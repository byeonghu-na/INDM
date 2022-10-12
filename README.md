# Maximum Likelihood Training of Implicit Nonlinear Diffusion Model (INDM) (NeurIPS 2022)

| [paper](https://arxiv.org/abs/2205.13699) | [pretrained model](https://www.dropbox.com/sh/yapgdylhkm4j0hu/AACh4jvT7wtBgVNmiUprRSdCa?dl=0) |


This repo contains an official PyTorch implementation for the paper "[Maximum Likelihood Training of Implicit Nonlinear Diffusion Model](https://arxiv.org/abs/2205.13699)" in [NeurIPS 2022](https://neurips.cc/Conferences/2022/).

**[Dongjun Kim](https://github.com/Kim-Dongjun) \*, [Byeonghu Na](https://github.com/byeonghu-na) \*, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, and Il-Chul Moon**   
<sup> * Equal contribution </sup>
 
--------------------

This paper introduces **Implicit Nonlinear Diffusion Model (INDM)**, that learns the nonlinear diffusion process by combining a normalizing flow and a diffusion process.

<img src="./figures/overview.png" width="1000" title="overview" alt="INDM attains a ladder structure between the data space and the latent space. The latent vector is visualized by normalizing the latent value">

## Requirements

This code was tested with CUDA 11.1 and Python 3.8.

```
pip install -r requirements.txt
```

## Pretrained Checkpoints

We release our checkpoints [here](https://www.dropbox.com/sh/yapgdylhkm4j0hu/AACh4jvT7wtBgVNmiUprRSdCa?dl=0).


## Stats files for FID evaluation

Download stats files and save it to `./assets/stats/`.

* For CIFAR-10, we use the stats file comes from [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch/blob/main/README.md#stats-files-for-quantitative-evaluation).
* For CelebA, we provide the stats file [`celeba_stats.npz`](https://www.dropbox.com/s/0y5c3q6qbehjxjx/celeba_stats.npz?dl=0).


## Training and Evaluation

* When you run a deep model, change the value of `model.num_res_blocks` from 4 to 8 in the config file. (or add `--config.model.num_res_blocks 8` in your script.) 

### CIFAR-10

#### INDM (VE, FID)

* Training
```
python main.py --mode train --config configs/ve/CIFAR10/indm.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/ve/CIFAR10/indm.py --workdir <work_dir> --config.sampling.pc_denoise=True --config.sampling.pc_denoise_time -1 --config.sampling.begin_snr 0.14 --config.sampling.end_snr 0.14 --config.eval.data_mean=True
```

#### INDM (VP, FID)

* Training
```
python main.py --mode train --config configs/vp/CIFAR10/indm_fid.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CIFAR10/indm_fid.py --workdir <work_dir> --config.sampling.temperature 1.05
```

#### INDM (VP, NLL)

* Training
```
python main.py --mode train --config configs/vp/CIFAR10/indm_nll.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CIFAR10/indm_nll.py --workdir <work_dir>
```

### CelebA

#### INDM (VE, FID)

* Training
```
python main.py --mode train --config configs/ve/CELEBA/indm.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/ve/CELEBA/indm.py --workdir <work_dir>
```

#### INDM (VP, FID)

* Training
```
python main.py --mode train --config configs/vp/CELEBA/indm_fid.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CELEBA/indm_fid.py --workdir <work_dir>
```

#### INDM (VP, NLL)

* Training
```
python main.py --mode train --config configs/vp/CELEBA/indm_nll.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CELEBA/indm_nll.py --workdir <work_dir>
```



## Acknowledgements

This work is heavily built upon the code from
* [Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." *International Conference on Learning Representations (ICLR)*. 2021.](https://github.com/yang-song/score_sde_pytorch)
* [Song, Yang, et al. "Maximum likelihood training of score-based diffusion models." *Advances in Neural Information Processing Systems (NeurIPS)*. 2021.](https://github.com/yang-song/score_flow)
* [Xuezhe Ma, et al. "Decoupling Global and Local Representations via Invertible Generative Flows." *International Conference on Learning Representations (ICLR)*. 2021.](https://github.com/XuezheMax/wolf)

## References
If you find the code useful for your research, please consider citing
```bib 
@inproceedings{kim2022maximum,
  title={Maximum Likelihood Training of Implicit Nonlinear Diffusion Model},
  author={Dongjun Kim and Byeonghu Na and Se Jung Kwon and Dongsoo Lee and Wanmo Kang and Il-Chul Moon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
 }
 ```
