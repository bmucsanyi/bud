import os

from .reader_image_folder import ReaderImageFolder
from .reader_image_in_tar import ReaderImageInTar


def create_reader(name, root, split="train", **kwargs):
    name = name.lower()
    name = name.split("/", 1)
    prefix = ""
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]

    # FIXME improve the selection right now just tfds prefix or fallback path, will need
    # options to explicitly select other options shortly
    if prefix == "hfds":
        from .reader_hfds import ReaderHfds  # defer tensorflow import

        reader = ReaderHfds(root, name, split=split, **kwargs)
    elif prefix == "tfds":
        from .reader_tfds import ReaderTfds  # defer tensorflow import

        reader = ReaderTfds(root, name, split=split, **kwargs)
    elif prefix == "wds":
        from .reader_wds import ReaderWds

        kwargs.pop("download", False)
        reader = ReaderWds(root, name, split=split, **kwargs)
    elif prefix == "soft":
        from .reader_soft import ReaderSoft

        dataset_name_to_path = {
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
        ds_path = dataset_name_to_path.get(name)
        assert ds_path is not None, f"Soft label dataset {name} is not implemented."
        ds_path = os.path.join(root, ds_path)

        reader = ReaderSoft(ds_path, split=split, **kwargs)
    elif prefix == "repr":
        if name == "cub":
            from .reader_repr import ReaderCUB

            reader = ReaderCUB(os.path.join(root, "cub200"), split=split, **kwargs)
        elif name == "cars":
            from .reader_repr import ReaderCARS

            reader = ReaderCARS(os.path.join(root, "cars196"), split=split, **kwargs)
        elif name == "sop":
            from .reader_repr import ReaderSOP

            reader = ReaderSOP(
                os.path.join(root, "online_products"), split=split, **kwargs
            )
    else:
        assert os.path.exists(root)
        # default fallback path (backwards compat), use image tar if root is a .tar file,
        # otherwise image folder
        # FIXME support split here or in reader?
        if os.path.isfile(root) and os.path.splitext(root)[1] == ".tar":
            reader = ReaderImageInTar(root, **kwargs)
        else:
            reader = ReaderImageFolder(root, **kwargs)
    return reader
