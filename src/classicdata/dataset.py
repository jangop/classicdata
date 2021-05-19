"""
Base classes for datasets.
"""
from abc import abstractmethod


class Dataset:
    """
    Abstract base class for datasets.
    """

    def __init__(self, safe_name: str, short_name: str, long_name: str):
        self.safe_name = safe_name
        self.short_name = short_name
        self.long_name = long_name

        self.loaded: bool = False

    @abstractmethod
    def load(self):
        """
        Expected to fill self.points and self.labels, fit self.label_encoder, and set self.loaded.
        """
        pass
