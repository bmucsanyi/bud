import re
from dataclasses import dataclass

from torch.nn import Module


@dataclass
class ModuleData:
    variable_name: str
    module_name: str
    module: Module


def deep_setattr(obj, attr_path, value):
    parts = attr_path.split(".")

    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    if parts[-1].isdigit():
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)


def replace(model: Module, source_regex: str, target_module: Module):
    source_regex = re.compile(source_regex)

    module_datas = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_datas = [
        module_data
        for module_data in module_datas
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_datas:
        deep_setattr(
            model,
            matched_module_data.variable_name,
            target_module(matched_module_data.module),
        )
