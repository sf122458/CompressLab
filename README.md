# CompressLab

A PyTorch-based framework for deep-learning-based data compression research.

### Features
- **Multi-λ Training & Parallel Model Execution**

  Support setting multiple λ values in the loss function to train multiple models simultaneously, enabling efficient hyperparameter tuning and multi-scenario model optimization. `TensorBoard` is used to monitor the training process.

- **Automatic Metric Calculation via Dataclass Encapsulation**

  Inputs and outputs of compression models are encapsulated in dataclass, which automatically computes key metrics based on input arguments. For example, providing `likelihoods` allows the dataclass to directly calculate the `bpp` (bits per pixel) attribute—simply access the attribute to obtain results without manual computation.

- **YAML-Driven Zero-Code Experiment Launch**

  Start experiments with a single `yaml` configuration file:
  - Model Auto-Registration: The `Registry` class enables automatic model registration. Specify a list of models under the `Model` key in the config file to train multiple models in one run.
  - DDP Training Simplification: Built on [PyTorch Lightning](https://lightning.ai), configure the `Devices` key in the config file to easily enable DDP (Distributed Data Parallel) training for multi-GPU acceleration.
  - Each config file responds to a folder in the `output` directory, so it is suggested to train models oriented to a specific task in a single config file.
  - **NOTE**: All trainers are defined in `module.py`. Models choose the corresponding trainer based on the `model` annotation defined in `LightningModule` class, so if you want to define a new model, you need to inherit from the abstract class in `abc.py`. The abstract class defines some common methods that all models should implement. You can refer to the existing models in `model` folder for implementation details.

- **Comprehensive Benchmarking Suite**
  
  The `Benchmark` module can automatically collect multiple metrics and write into `metric.txt`, including:
  - **Quality Metrics**: Bits per pixel (bpp), peak signal-to-noise ratio (PSNR);
  - **Performance Metrics**: Compression/decompression speed;
  - **Comparative Metrics**: BD-rate across models for direct performance comparison.


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

Each folder in `dataset` has a `download.sh` script to download the dataset. Just run `download.sh` and don't need to care about the file structure. Currently, the following datasets are supported:
- `CLIC 2020` is used for the training of the image compression models.
- `Kodak` is used for the evaluation of the image compression models.
- `Vimeo-90K` is used for the training of the video compression models.
- `UVG` is used for the evaluation of the video compression models.

### Available models
#### Lossy image compression
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

#### Video compression
- [DVC](https://arxiv.org/abs/1812.00101)
- [SSF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf)

**Note: All models can be trained using the provided code, but the compression performance remains unverified.**


### Related links
- [PyTorch Lightning](https://lightning.ai)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [ChARM-PyTorch](https://github.com/tokkiwa/minnen2020)
- [LIC-TCM](https://github.com/jmliu206/LIC_TCM)
- [MLIC](https://github.com/JiangWeibeta/MLIC)
- [DVC-PyTorch](https://github.com/binzzheng/DVC-PyTorch)
- [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression)
- [CLIC](https://www.compression.cc/)
- [UVG](https://ultravideo.fi)
- [Kodak](https://r0k.us/graphics/kodak/)