"""This file contains utils and a dataloader to load the downstream datasets
of the "Is One Annotation Enough?" paper

Hacked together by / Copyright 2023 Michael Kirchhof
                           and 2024 Bálint Mucsányi
"""

import json
import os

import numpy as np
from torchvision.datasets.folder import pil_loader

from .img_extensions import get_img_extensions
from .reader import Reader


class ReaderSoft(Reader):
    def __init__(
        self,
        root,
        is_evaluate_on_all_splits_id=False,
        split="test",
        fold_idx=0,
        **kwargs,
    ):
        """
        root: string, path to root directory of dataset
        split: string, train/validation/test/all. Which folds to use
        fold_idx: integer 0, 1, 2, 3, 4. Which 5-fold crossvalidation to use. Default 0
            (same as in "is one annotation enough")
        """
        super().__init__()

        # Load the soft labels
        (
            self.soft_labels,
            self.class_to_idx,
            self.filepath_to_imgid,
        ) = self.load_raw_annotations(os.path.join(root, "annotations.json"))
        self.soft_labels = np.concatenate(
            [self.soft_labels, self.soft_labels.argmax(axis=-1, keepdims=True)], axis=-1
        )

        self.root = os.path.split(root)[0]
        self.samples = self.filepath_to_imgid.keys()

        # Restrict self.samples to val/test
        current_folds = []
        if split == "validation":
            if is_evaluate_on_all_splits_id:
                current_folds = [
                    f"fold{(0 + fold_idx) % 5 + 1}",
                    f"fold{(1 + fold_idx) % 5 + 1}",
                    f"fold{(2 + fold_idx) % 5 + 1}",
                    f"fold{(3 + fold_idx) % 5 + 1}",
                    f"fold{(4 + fold_idx) % 5 + 1}",
                ]
            else:
                current_folds = [
                    f"fold{(0 + fold_idx) % 5 + 1}",
                    f"fold{(1 + fold_idx) % 5 + 1}",
                ]
        elif split == "test":
            current_folds = [
                f"fold{(2 + fold_idx) % 5 + 1}",
                f"fold{(3 + fold_idx) % 5 + 1}",
                f"fold{(4 + fold_idx) % 5 + 1}",
            ]
        elif split == "all":
            current_folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        self.samples = [s for s in self.samples if any(f in s for f in current_folds)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {root}. "
                f'Supported image extensions are {", ".join(get_img_extensions())}'
            )

    def __getitem__(self, index):
        """
        Return format is
        1) PIL image
        2) array where first column is target label and remaining columns are raw label
           counts
        """
        path = self.samples[index]
        soft_labels = self.soft_labels[self.filepath_to_imgid[path], :]
        path = os.path.join(self.root, path)
        return pil_loader(path), soft_labels

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    @staticmethod
    def load_raw_annotations(path):
        """
        Casts the raw annotations from an annotations.json into a numpy array of label
        votes per image.
        """

        with open(path) as f:
            raw = json.load(f)

            # Collect all annotations
            img_filepath = []
            labels = []
            for annotator in raw:
                for entry in annotator["annotations"]:
                    # Add only valid annotations to table
                    if (label := entry["class_label"]) is not None:
                        img_filepath.append(entry["image_path"])
                        labels.append(label)

            # Summarize the annotations
            unique_img_filepath = list(np.unique(np.array(img_filepath)))
            filepath_to_imgid = dict(
                zip(unique_img_filepath, list(np.arange(0, len(unique_img_filepath))))
            )
            unique_labels = list(np.unique(np.array(labels)))
            classname_to_labelid = dict(
                zip(unique_labels, list(np.arange(0, len(unique_labels))))
            )
            soft_labels = np.zeros(
                (len(unique_img_filepath), len(unique_labels)), dtype=np.int64
            )
            for filepath, classname in zip(img_filepath, labels):
                soft_labels[
                    filepath_to_imgid[filepath], classname_to_labelid[classname]
                ] += 1

            return soft_labels, classname_to_labelid, filepath_to_imgid
