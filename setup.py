"""Loaders for classic datasets commonly used in Machine Learning.

See:
https://github.com/jangop/classicdata
"""

import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="classicdata",
    version="0.1.0-alpha1",
    description="Loaders for classic datasets commonly used in Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangop/classicdata",
    author="Jan Philip Göpfert",
    author_email="janphilip@gopfert.eu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="machine learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9, <4",
    install_requires=["loguru", "scikit-learn", "numpy", "appdirs"],
    extras_require={
        "dev": ["check-manifest", "black", "pylint"],
        "test": ["coverage", "pytest", "black", "pylint"],
    },
    project_urls={
        "Bug Reports": "https://github.com/jangop/classicdata/issues",
        "Source": "https://github.com/jangop/classicdata",
    },
)
