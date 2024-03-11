""" Misc utils

Hacked together by / Copyright 2020 Ross Wightman
                           and 2024 Bálint Mucsányi
"""
import argparse
import ast
import importlib
import re

import matplotlib.pyplot as plt
import torch
from torch import nn


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def add_bool_arg(parser, name, default=False, help=""):
    dest_name = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=dest_name, action="store_true", help=help)
    group.add_argument("--no-" + name, dest=dest_name, action="store_false", help=help)
    parser.set_defaults(**{dest_name: default})


# class ParseKwargs(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         kw = {}
#         for value in values:
#             key, value = value.split("=")
#             try:
#                 kw[key] = ast.literal_eval(value)
#             except ValueError:
#                 kw[key] = str(
#                     value
#                 )  # fallback to string (avoid need to escape on command line)
#         setattr(namespace, self.dest, kw)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        # Split the string on spaces to get each key-value pair
        pairs = values.split(" ")
        for pair in pairs:
            # Split each pair on "=" to separate keys and values
            key, value = pair.split("=", 1)
            try:
                # Attempt to parse the value into a Python object
                kw[key] = ast.literal_eval(value)
            except ValueError:
                # Keep the value as a string if parsing fails
                kw[key] = value
        setattr(namespace, self.dest, kw)


def type_from_string(type_string):
    if type_string is None:
        return type_string

    # Split the string into module and type
    module_name, type_name = type_string.rsplit(".", 1)

    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Get the type from the module
        return getattr(module, type_name)
    except (ImportError, AttributeError) as e:
        raise argparse.ArgumentTypeError(f"Could not import type {type_string}: {e}")


def extract_layer_candidates(model, layertype=nn.ReLU):
    relu_layers = []

    for name, module in model.named_modules():
        if isinstance(module, layertype):
            relu_layers.append((name, module))

    return relu_layers


def show_image_grid(img_tensors, mean, std, grid_size=(3, 3)):
    _, axes = plt.subplots(*grid_size, figsize=(12, 12))  # Adjust figsize as needed
    axes = axes.flatten()

    for idx, img_tensor in enumerate(img_tensors):
        if idx >= grid_size[0] * grid_size[1]:
            break  # Stop if we have more images than grid slots

        img_tensor = img_tensor.permute(1, 2, 0) * std + mean
        img_tensor = torch.minimum(
            torch.ones(1), torch.maximum(torch.zeros(1), img_tensor)
        )

        axes[idx].imshow(img_tensor)
        axes[idx].axis("off")  # Turn off axis

    plt.show()
