# CompressLab

A PyTorch-based framework for deep-learning-based data compression research.

### Features

- All implementations of models and datasets are modularized and registered in a registry. The training and testing pipelines can be easily configured by YAML files. Please refer to the example configuration files in the `config` folder and see the details of available options in `compresslab/utils/config.py`.
- For the vae-based image compression models `compresslab/nn/lossy_image_compression`, training with several rate-distortion points simultaneously is supported in one run. 
- The framework is built on [PyTorch Lightning](https://lightning.ai), which makes it easy to run in DDP mode and adjust the precision. Moreover, it supports logging via TensorBoard, and resume from the checkpoint automatically.
- A general metrics computation module is provided, and you can easily obtain the final results in the `output` folder after testing.

### Installation

```shell
conda create -n compresslab python=3.9
conda activate compresslab
pip install -e .
```

### How to use

Run
```shell
python run.py --list
```
to check all registered modules.

Run
```shell
python run.py --config config/your_config.yaml
```
to train a model with the specified configuration file.

Run
```shell
tensorboard --logdir output
```
to monitor the training process.

### Dataset preparation

`download.py` can be used to download and extract the datasets automatically. For example, to download the CLIC2020 Professional Validation dataset, run
```shell
python dataset/download.py --dataset CLIC2020_professional_val
```
Current supported datasets:
- Vimeo90k
- CLIC2020
- DIV2K
- Tecnick
- UVG
- Kodak (Already included in the repository)

### Available models
#### VAE-based lossy image compression
- All models implemented in [CompressAI](https://github.com/InterDigitalInc/CompressAI):
  - [FactorizedPrior](https://arxiv.org/abs/1607.05006)
  - [ScaleHyperprior](https://arxiv.org/abs/1802.01436)
  - [MeanScaleHyperprior, JointAutoregressiveandHierarchicalPriors](https://arxiv.org/abs/1809.02736)
  - [Cheng2020](https://arxiv.org/abs/2001.01568)
  - [Cheng2020Checkerboard](https://arxiv.org/abs/2103.15306)
  - [ELIC](https://arxiv.org/abs/2203.10886)
- [ChARM](https://arxiv.org/abs/2007.08739)
- [TCM](https://arxiv.org/abs/2303.14978)
- [MLIC](https://arxiv.org/abs/2307.15421)

#### Generative image compression (Only support evaluation of the pre-trained models now)
- [DiffEIC](https://arxiv.org/pdf/2404.18820)
- [StableCodec](https://arxiv.org/abs/2506.21977)

<!-- #### Video compression (TODO: refactor the code)
- [DVC](https://arxiv.org/abs/1812.00101)
- [SSF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf)
 -->


### TODO
- [ ] Refactor the code of video compression models.
- [ ] Load video dataset directly from `.yuv` files.
- [ ] Try to implement finetining from checkpoints or two-stage training in one run.
- [ ] Support automatically padding the input image to the required size.
- [ ] `benchmark.py` can't give the correct bd-rate curve. Fix it later.


### Related links
- [PyTorch Lightning](https://lightning.ai)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [ChARM-PyTorch](https://github.com/tokkiwa/minnen2020)
- [LIC-TCM](https://github.com/jmliu206/LIC_TCM)
- [MLIC](https://github.com/JiangWeibeta/MLIC)
- [DVC-PyTorch](https://github.com/binzzheng/DVC-PyTorch)
- [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression)
- [DiffEIC](https://github.com/huai-chang/DiffEIC)
- [StableCodec](https://github.com/LuizScarlet/StableCodec)
- [CLIC](https://www.compression.cc/)
- [UVG](https://ultravideo.fi)
- [Kodak](https://r0k.us/graphics/kodak/)