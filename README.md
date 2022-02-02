# Classic Datasets

Loaders for classic datasets commonly used in Machine Learning:

| Name                                              |   # Samples |   # Features |   # Classes |
|---------------------------------------------------|-------------|--------------|-------------|
| [Ionosphere](https://archive.ics.uci.edu)         |         351 |           34 |           2 |
| [Letter Recognition](https://archive.ics.uci.edu) |       20000 |           16 |          10 |
| [Telescope](https://archive.ics.uci.edu)          |       19020 |           10 |           2 |
| [Pen Digits](https://archive.ics.uci.edu)         |       10992 |           16 |          10 |
| [Robot Navigation](https://archive.ics.uci.edu)   |        5456 |           24 |           4 |
| [Segmentation](https://archive.ics.uci.edu)       |        2310 |           16 |           7 |
| [USPS](http://www.gaussianprocess.org/gpml/data/) |        9298 |          256 |          10 |

## Installation

```
pip install classicdata
```

Run `python -m classicdata.info` to list all implemented datasets.

## Example Usage

```python
from classicdata import Ionosphere

ionosphere = Ionosphere()

# Use ionosphere.points and ionosphere.labels...
```