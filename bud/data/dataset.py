""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019 Ross Wightman
                           and 2024 Bálint Mucsányi
"""

import json
import logging
import os
from typing import Any, Optional, Tuple
from torchvision.datasets.folder import pil_loader
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from .imagenet import ImageNet

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50

DATASET_NAME_TO_PATH = {
    # Closest datasets to ImageNet (containing natural objects)
    "cifar10": "CIFAR10H",
    "treeversity1": "Treeversity#1",
    "turkey": "Turkey",
    "pig": "Pig",
    "benthic": "Benthic",
    # Medical datasets (bit larger shift from pretraining)
    "micebone": "MiceBone",
    "planktion": "Planktion",
    "qualitymri": "QualityMRI",
    # Synthetic dataset, currently unused
    "synthetic": "Synthetic",
    # Same as Treeversity#6, but with 6 tags per image instead of one class
    # (doesn't make sense to use both Treeversity#1 and Treeversity#6)
    "treeversity6": "Treeversity#6",
}


class ImageDataset(data.Dataset):
    def __init__(
        self,
        root,
        reader=None,
        split="train",
        class_map=None,
        load_bytes=False,
        img_mode="RGB",
        transform=None,
        target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or "", root=root, split=split, class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(
                f"Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}"
            )
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):
    def __init__(
        self,
        root,
        reader=None,
        split="train",
        class_map=None,
        is_training=False,
        batch_size=None,
        seed=42,
        repeats=0,
        download=False,
        transform=None,
        target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.is_training = is_training
        self.split = split
        self.is_ood = False

    def __iter__(self):
        if self.is_ood:
            rng = np.random.default_rng(seed=0)
        else:
            rng = None

        for img, target in self.reader:
            if self.transform is not None and self.is_ood:
                img = self.transform(img, rng)
            elif self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def set_ood(self):
        self.is_ood = True

    def __len__(self):
        if hasattr(self.reader, "__len__"):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, "set_epoch"):
            self.reader.set_epoch(count)

    def set_loader_cfg(
        self,
        num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need #workers for correct #samples estimate before loader
        # processes created
        if hasattr(self.reader, "set_loader_cfg"):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, "Filename lookup by index not supported, use filenames()."

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert (
            isinstance(x, (list, tuple)) and len(x) == 3
        ), "Expecting a tuple/list of 3 transforms"
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # All splits share the same dataset base transform
        x_list = [
            self._normalize(x)
        ]  # First split only normalizes (this is the 'clean' split)
        # Run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


class SoftImageNet(ImageNet):
    def __init__(
        self, root: str, label_root: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(root, split="val", **kwargs)

        if label_root is None:
            label_root = root

        self.soft_labels, self.filepath_to_softid = self.load_raw_annotations(
            path_soft_labels=os.path.join(label_root, "raters.npz"),
            path_real_labels=os.path.join(label_root, "real.json"),
        )

        self.is_ood = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, original_target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        converted_index = self.filepath_to_softid[
            os.path.split(self.samples[index][0])[-1]
        ]
        soft_target = self.soft_labels[converted_index, :]
        augmented_target = np.concatenate([soft_target, [original_target]])

        if self.target_transform is not None:
            augmented_target = self.target_transform(augmented_target)

        return img, augmented_target

    def set_ood(self):
        self.is_ood = True

    @staticmethod
    def load_raw_annotations(path_soft_labels, path_real_labels):
        # Loads the raw annotations from raters.npz from github.com/google-research/reassessed-imagenet
        # Adapted from github.com/google/uncertainty-baselines/blob/main/baselines/jft/data_uncertainty_utils.py#L87
        data = np.load(path_soft_labels)

        summed_ratings = np.sum(data["tensor"], axis=0)  # 0 is the annotator axis
        yes_prob = summed_ratings[:, 2]
        # This gives a [questions] np array.
        # It gives how often the question "Is image X of class Y" was answered with "yes".

        # We now need to summarize these questions across the images and labels
        num_labels = 1000
        soft_labels = {}
        for idx, (file_name, label_id) in enumerate(data["info"]):
            if file_name not in soft_labels:
                soft_labels[file_name] = np.zeros(num_labels, dtype=np.int64)
            added_label = np.zeros(num_labels, dtype=np.int64)
            added_label[int(label_id)] = yes_prob[idx]
            soft_labels[file_name] = soft_labels[file_name] + added_label

        # Questions were only asked about 24889 images, and of those 1067 have no single yes vote at any label
        # We will fill up (some of) the missing ones by taking the ImageNet Real Labels
        new_soft_labels = {}
        with open(path_real_labels) as f:
            real_labels = json.load(f)
        for idx, label in enumerate(real_labels):
            key = "ILSVRC2012_val_"
            key += (8 - len(str(idx + 1))) * "0" + str(idx + 1) + ".JPEG"
            if len(label) > 0:
                one_hot_label = np.zeros(num_labels, dtype=np.int64)
                one_hot_label[label] = 1
                new_soft_labels[key] = one_hot_label
            else:
                new_soft_labels[key] = np.zeros(num_labels)

        # Merge soft and hard labels
        unique_img_filepath = list(new_soft_labels.keys())
        filepath_to_imgid = dict(
            zip(unique_img_filepath, list(np.arange(0, len(unique_img_filepath))))
        )
        soft_labels_array = np.zeros((len(unique_img_filepath), 1000), dtype=np.int64)
        for idx, img in enumerate(unique_img_filepath):
            if img in soft_labels and soft_labels[img].sum() > 0:
                final_soft_label = soft_labels[img]
            else:
                final_soft_label = new_soft_labels[img]
            soft_labels_array[idx, :] = final_soft_label

        # Note that 750 of the 50000 images in soft_labels_array will still not have a
        # label at all. These are ones where the old imagenet label was false and also
        # the raters could not determine any new one. We hand 0 matrices out for them.
        # They should be ignored in computing the metrics

        return soft_labels_array, filepath_to_imgid

class SoftDataset(data.Dataset):
    def __init__(
        self,
        name,
        root,
        split="train",
        is_training=False,
    ):
        ds_path = DATASET_NAME_TO_PATH[name]
        root = os.path.join(root, ds_path)

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
            current_folds = [
                f"fold{i}" for i in range(1, 3)
            ]
        elif split == "test":
            current_folds = [
                f"fold{i}" for i in range(3, 6)
            ]
        elif split == "all":
            current_folds = [
                f"fold{i}" for i in range(1, 6)
            ]
        self.samples = [s for s in self.samples if any(f in s for f in current_folds)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {root}."
            )

        self.transform = None
        self.target_transform = None
        self._consecutive_errors = 0
        self.is_training = is_training
        self.split = split
        self.is_ood = False

    def __getitem__(self, index):
        """
        Return format is
        1) PIL image
        2) array where first column is target label and remaining columns are raw label
           counts
        """
        path = self.samples[index]
        path = os.path.join(self.root, path)

        target = self.soft_labels[self.filepath_to_imgid[path], :]
        img = pil_loader(path)
        
        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def set_ood(self):
        self.is_ood = True

    def __len__(self):
        return len(self.samples)

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
