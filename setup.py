from setuptools import setup, find_packages

setup(
    name='compresslab',
    version='0.1',
    packages=find_packages(),
    description="Pytorch training framework",
    install_requires=[
        'torch',
        'torchvision',
        # 'pytorch-lightning',
        'pyyaml',
        'tqdm',
        'numpy',
        'wandb',
        # 'pydantic',
        'marshmallow',
        # 'marshmallow_dataclass',
        'dataclasses',
        # 'yacs',
        # 'pytorch-msssim',
        # 'torch-ssim',
        # 'torchmetrics'
    ],
    python_requires='>=3.7',
)