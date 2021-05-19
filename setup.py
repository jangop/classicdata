"""Loaders for classic datasets commonly used in Machine Learning.

See:
https://github.com/jangop/classic-data
"""

import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="classicdata",
    version="0.1.0-alpha",
    description="Loaders for classic datasets commonly used in Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangop/classic-data",
    author="Jan Philip Göpfert",
    author_email="janphilip@gopfert.eu",  # Optional
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Machine Learning",
        "License :: The Unlicense",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="machine learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9, <4",
    install_requires=["loguru", "scikit-learn", "numpy", "appdirs"],
    extras_require={
        "dev": ["check-manifest", "black"],
        "test": ["coverage", "pytest", "black"],
    },
    project_urls={
        "Bug Reports": "https://github.com/jangop/classic-data/issues",
        "Source": "https://github.com/jangop/classic-data",
    },
)
