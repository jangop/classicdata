# Classic Datasets

Loaders for classic datasets commonly used in Machine Learning:

| Dataset                                           |   # Samples |   # Features | Feature Type   |   # Classes |   Balance |
|---------------------------------------------------|-------------|--------------|----------------|-------------|-----------|
| [Ionosphere](https://archive.ics.uci.edu)         |         351 |           34 | unknown        |           2 |      0.56 |
| [Letter Recognition](https://archive.ics.uci.edu) |       20000 |           16 | unknown        |          10 |      0.91 |
| [Telescope](https://archive.ics.uci.edu)          |       19020 |           10 | unknown        |           2 |      0.54 |
| [Pen Digits](https://archive.ics.uci.edu)         |       10992 |           16 | unknown        |          10 |      0.92 |
| [Robot Navigation](https://archive.ics.uci.edu)   |        5456 |           24 | numerical      |           4 |      0.15 |
| [Segmentation](https://archive.ics.uci.edu)       |        2310 |           16 | numerical      |           7 |      1.00 |
| [USPS](http://www.gaussianprocess.org/gpml/data/) |        9298 |          256 | numerical      |          10 |      0.46 |

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

## Related Projects

There are other projects. They are more mature, more robust, more better.
That is why this project is called **classic**data. Sometimes you need small, simple datasets.
Other times, consider the following projects.

- [OpenML](https://www.openml.org/): better, faster, stronger; more complex, though
- [sklearn.datasets](https://scikit-learn.org/stable/datasets.html): limited selection; no metadata
- [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html): limited selection; datasets too modern (big)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets): datasets too modern (big)