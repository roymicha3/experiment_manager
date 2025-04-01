from setuptools import setup, find_packages

setup(
    name="experiment_manager",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "omegaconf>=2.3.0",
    ],
    author="Roy Michaeli",
    description="A Python package for managing experiments",
    python_requires=">=3.6",
)
