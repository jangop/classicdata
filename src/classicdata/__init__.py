"""
Loaders for classic datasets.
"""
from .datasets.ionosphere import Ionosphere
from .datasets.letterrecognition import LetterRecognition
from .datasets.magic import MagicGammaTelescope
from .datasets.pendigits import PenDigits
from .datasets.robnav import RobotNavigation
from .datasets.segmentation import ImageSegmentation

__all__ = [
    "Ionosphere",
    "LetterRecognition",
    "MagicGammaTelescope",
    "PenDigits",
    "RobotNavigation",
    "ImageSegmentation",
]
