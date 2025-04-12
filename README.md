# CompressLab

This repository is based on further encapsulation of [PyTorch Lightning](https://lightning.ai).

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
- [ ] Implementations on compress and decompress.
- [ ] Benchmark test, such as speed, model size, etc.