"""
setup.py -- Package installation config for xray-gradcam.

Install in editable mode for development:
    pip install -e .

This makes 'from src.model import ...' work from any directory without
manually managing sys.path in every script or test file.
"""

from setuptools import find_packages, setup

setup(
    name="xray-gradcam",
    version="0.1.0",
    description=(
        "Grad-CAM explainability for chest X-ray pneumonia detection "
        "using DenseNet121"
    ),
    author="LSaiko",
    url="https://github.com/LSaiko/xray-gradcam",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "grad-cam>=1.4.8",
        "pydicom>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
