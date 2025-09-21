CACHE_DIR = "cache"
PRETRAINED_CACHE_DIR = f"{CACHE_DIR}/pretrained_models" # The directory to cache pretrained models' checkpoint.
OUTPUT_DIR = "output"               # The output directory for all experiments.
MODEL_DEFAULT_FILENAME = "models.py"    # The model filename. The code will automatically register models in this file.
TRAINER_DEFAULT_FILENAME = "trainer.py" # The trainer filename. The code will automatically search the trainer in this file.


# logging message in terminal
LOGGING_SHOW_ABS_PATH = False # If true, show absolute path in terminal (friendly to locate the file in VSCode).

# tensorboard logging
LOGGING_EVERY_N_STEP = 1000 # Log every n steps in training.