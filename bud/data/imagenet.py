"""Minimal ImageNet dataset that only requires train and val folders

Hacked together by / Copyright 2024 Anonymous Author
"""

import os
from typing import Any

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class ImageNet(ImageFolder):
    """Minimal ImageNet <http://image-net.org/> 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        super().__init__(self.split_folder, **kwargs)
        self.root = root

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
