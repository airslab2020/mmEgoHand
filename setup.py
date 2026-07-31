"""Compatibility shim for editable installs with older pip versions."""

from setuptools import find_packages, setup


setup(
    name="mmegohand",
    version="1.0.0",
    packages=find_packages(include=("mmegohand", "mmegohand.*")),
    python_requires=">=3.9",
    install_requires=[
        "h5py>=3.8",
        "numpy>=1.23",
        "psutil>=5.9",
        "PyYAML>=6.0",
        "scipy>=1.9",
        "torch>=1.13",
    ],
)
