The config file is a `.yaml` file. You can refer to the `template.yaml` for the format of the config file.

# model
- `compound`: Choose the compound of the model. The compound is responsible for the collect the output of the model and calculate the loss.
- `net`: A dictionary that contains the parameters for the model. 
    In the dictionary, the key is the name of the model(you can define it by yourself), and the value is a dictionary that contains the key and parameters for the model. Here the key is the registered name of the model and the params are the parameters for the initialization of the model.


# train
- `trainer`: Choose the trainer for the model. The trainer deals with the preparation of dataset(you can define the dataset format and use it in the trainer), model and optimizer, recording necessary data, the progress bar and so on.
- `batchsize`: Batch size for training.
- `epochs`: Number of epochs for training.
- `valinterval`: Interval of validation.
- `trainset`: 
    - `path`: Path to the training dataset.
    - `transform`: Transform for the training dataset. 
- `valset`: Path to the validation dataset.
    - `path`: Path to the validating dataset.
    - `transform`: Transform for the vakidating dataset.
- `output`: Path to save the output of the model. By default, a folder named "output" will be created and the folder with be saved in it. 
- `loss`: Loss function for the model. It should be a dictionary where the key is the name of the loss function registered in the `loss` module and the value is the weight of the loss function.
-  `optim`: Optimizer for the model. The `key` is the name of the optimizer registered in the `optim` module and the `params` are the parameters for the optimizer same with Pytorch, such as `lr`.
- `schdr`(Optional): Scheduler for the optimizer. Its format is the same with `optim`.

# log
- `key`: `wandb` or `tensorboard` or `None`. Default is `None`, which means no logger will be used.
- `params`: Parameters for the logger. 


# env
- `WANDB_APT_KEY`: API key for Weights & Biases, used when wandb is enabled.
- `CUDA_VISIBLE_DEVICES`(Not implemented): GPU device to use. Default is `0`.
- `NUM_WORKERS`: Number of workers for dataloader. Default is `4`.