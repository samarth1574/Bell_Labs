"""
Setup script for the High-Density Object Segmentation package.
"""

from setuptools import setup, find_packages

setup(
    name="high-density-segmentation",
    version="0.1.0",
    description=(
        "Instance-level detection and segmentation of 1-50 heavily "
        "overlapping objects using Soft-NMS"
    ),
    author="Mrityunjay Singh",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "pycocotools>=2.0.7",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "isort"],
        "notebook": ["jupyter", "ipykernel"],
    },
)
