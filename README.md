# CompressLab
### Introduction
This is a framework for the training of Pytorch models, especially for models used in compression.

The config of all experiments can be set in `config/*.yaml`. Please refer to `config/template.yaml` for detailed information.

### Requirements

```bash
conda create -n compresslab python=3.7
pip install .
```


### How to use

```
output
├─Expname
    ├─Model1
        ├─ckpt
        ├─wandb
    ├─Model2
    ...
    ├─config.yaml
```

Run
```bash
python run_benchmark.py --config config/template.yaml
```
to start training.

To resume training from the checkpoint, you can use the original `.yaml` file or the copied `.yaml` file in the output path and run the same command mentioned above.


Run the following command to list all registered modules.
```bash
python run_benchmark.py --list
```
This will list all registered modules and show where they are registered. You can add other modules following the instruction in `compresslab/utils/registry.py`.


### TODO
- [x] Config interface.
- [x] Optimizer registration.
- [x] Loss function interface.
- [x] Dataset implementation.
- [x] Progress bar.
- [x] Trainer interface.
- [x] Breakpoint training.
- [ ] CompressAI Vbr model.
- [ ] VQ-VAE model.
- [ ] Video compression model: DVC, etc.
- [ ] Wandb and Tensorboard.
- [ ] DDP training.
- [ ] Notes for `.yaml` files.
- [ ] Metrics: speed, BD-rate, etc.