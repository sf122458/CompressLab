# CompressLab
### Introduction
This is a framework for the training of Pytorch models, especially for models used in compression.


<!-- `registry` is used to dynamically initialize all classes from the config in `yaml` files. See `utils/registry.py` for more information.  -->


The config of all experiments can be set in `config/*.yaml`. Please refer to `config/template.yaml` for detailed information.

### Requirements

```bash
conda create -n compresslab python=3.7
pip install requirements.txt
pip install .
```


### How to use

Run
```bash
python run_benchmark.py --config config/template.yaml
```
to start training.

Run the following command to list all registered modules.
```bash
python run_benchmark.py --list
```



### TODO
- [x] Config interface.
- [x] Optimizer registration.
- [x] Loss function interface.
- [x] Dataset implementation.
- [ ] Trainer interface.
- [ ] 
- [ ] Breakpoint training.
- [ ] Wandb and Tensorboard. What to record in trainer.
- [ ] DDP training.
- [ ] Notes for `.yaml` files.