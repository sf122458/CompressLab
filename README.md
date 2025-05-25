# CompressLab

A PyTorch-based framework for image compression research and development.

### Installation

```shell
conda create -n compresslab python=3.9
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

### Dataset preparation

Each folder in `dataset` has a `download.sh` script to download the dataset. Currently, the following datasets are supported:
- `CLIC 2020` is used for the training of the image compression models.
- `Kodak` is used for the evaluation of the image compression models.
- `Vimeo-90K` is used for the training of the video compression models.
- `UVG` is used for the evaluation of the video compression models.

### Support models
- All models implemented in [CompressAI](https://github.com/InterDigitalInc/CompressAI).
- MLIC
- DVC(debugging)


### TODO
- [x] Implementations on compress and decompress.
- [ ] Benchmark test, such as speed, model size, etc.
- [ ] Video compression framework.


### Related links
- [PyTorch Lightning](https://lightning.ai)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [MLIC++](https://github.com/JiangWeibeta/MLIC)
- [DVC-Pytorch](https://github.com/binzzheng/DVC-PyTorch)
- [PytorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression)
- [CLIC](https://www.compression.cc/)
- [UVG](https://ultravideo.fi)
- [Kodak](https://r0k.us/graphics/kodak/)