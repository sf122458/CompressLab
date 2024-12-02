from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent

with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='compresslab',
    version='0.1',
    packages=find_packages(),
    description="Pytorch training framework",
    install_requires=dependencies,
    python_requires='>=3.7',
)