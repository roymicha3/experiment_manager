from setuptools import setup, find_packages

setup(
    name="experiment_manager",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "omegaconf>=2.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pytest>=7.0.0",
        "typing_extensions>=4.5.0",
    ],
    author="Roy Michaeli",
    description="A Python package for managing experiments",
    python_requires=">=3.6",
)
