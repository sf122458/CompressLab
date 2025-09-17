import os
from pathlib import Path

try:
    import pybind11
except:
    os.system("pip install pybind11")

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension
from glob import glob

cwd = Path(__file__).resolve().parent

package_name = "compresslab"
version = "0.1.0"


def get_extensions():
    ext_modules = []

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=glob("compresslab/core/cpp_exts/rans/*.cpp"),
            language="c++",
            include_dirs=["compresslab/core/cpp_exts/rans"],
            extra_compile_args=extra_compile_args,
        )
    )

    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=glob("compresslab/core/cpp_exts/ops/*.cpp"),
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules

setup(
    name=package_name,
    version=version,
    python_requires=">=3.9",
    install_requires=[
        "einops",
        "lightning==2.5.1",
        "matplotlib",
        "scipy",
        "numpy",
        "opencv_python",
        "wget",
        "Pillow",
        "pydantic",
        "pytorch_msssim",
        "PyYAML",
        "rich==14.0.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchmetrics==1.8.1",
        "tensorboard",
        "transformers==4.37.2",
        "diffusers==0.35.1",
        "vector_quantize_pytorch",
        "open-clip-torch==2.22.0",
        "openai-clip=1.0.1",
        "peft==0.17.0",
        "setuptools_rust",
        "cupy-cuda12x",
        # "ema-pytorch==0.7.7"
    ],
    ext_modules=get_extensions(),
    rust_extensions=[
        RustExtension(
            f"compresslab.zipf_encoding",
            path="compresslab/nn/generative_image_compression/DiffC/rcc/arithmetic-coding/python-bindings/Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
