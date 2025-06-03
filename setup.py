from setuptools import setup, find_packages

setup(
    name="core",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0.0",
    ],
    python_requires=">=3.9",
    author="Kristophor Jensen",
    author_email="your.email@example.com",
    description="Core library for arc detection and data processing",
    long_description="Core library providing fundamental components for processing electrical arc data, including data loading, processing, metrics calculation, and visualization.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)