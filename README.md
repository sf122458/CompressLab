# CompressLab

A PyTorch-based framework for image compression research and development.

### Installation

```shell
conda create -n compresslab python=3.7
conda activate compresslab
pip install -r requirements.txt
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


### TODO
- [ ] BD-Rate calculation and plot.
- [x] Implementations on compress and decompress.
- [ ] Benchmark test, such as speed, model size, etc.


### Support models
- All models implemented in CompressAI.
- MLIC
- DVC(debugging)


### Acknowledgement
This repository is based on the following projects:
- [PyTorch Lightning](https://lightning.ai)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [MLIC++](https://github.com/JiangWeibeta/MLIC)
- [DVC-Pytorch](https://github.com/binzzheng/DVC-PyTorch)
- [PytorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression)