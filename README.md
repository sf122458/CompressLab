# CompressLab

A PyTorch-Based Framework for Data Compression Research.

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

### Dataset preparation

Each folder in `dataset` has a `download.sh` script to download the dataset. Currently, the following datasets are supported:
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