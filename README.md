# Classic Datasets

Loaders for classic datasets commonly used in Machine Learning:

| Dataset                                           |   # Samples |   # Features |   # Classes |   Balance |
|---------------------------------------------------|-------------|--------------|-------------|-----------|
| [Ionosphere](https://archive.ics.uci.edu)         |         351 |           34 |           2 |      0.56 |
| [Letter Recognition](https://archive.ics.uci.edu) |       20000 |           16 |          10 |      0.91 |
| [Telescope](https://archive.ics.uci.edu)          |       19020 |           10 |           2 |      0.54 |
| [Pen Digits](https://archive.ics.uci.edu)         |       10992 |           16 |          10 |      0.92 |
| [Robot Navigation](https://archive.ics.uci.edu)   |        5456 |           24 |           4 |      0.15 |
| [Segmentation](https://archive.ics.uci.edu)       |        2310 |           16 |           7 |      1.00 |
| [USPS](http://www.gaussianprocess.org/gpml/data/) |        9298 |          256 |          10 |      0.46 |

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