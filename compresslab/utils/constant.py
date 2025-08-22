CACHE_DIR = "cache"
PRETRAINED_CACHE_DIR = f"{CACHE_DIR}/pretrained_models" # The directory to cache pretrained models' checkpoint.
OUTPUT_DIR = "output"               # The output directory for all experiments.
MODEL_DEFAULT_FILENAME = "models.py"    # The model filename. The code will automatically register models in this file.
TRAINER_DEFAULT_FILENAME = "trainer.py" # The trainer filename. The code will automatically search the trainer in this file.


# logging
SHOW_ABS_PATH = False # If true, show absolute path in logging (friendly to locate the file in VSCode).