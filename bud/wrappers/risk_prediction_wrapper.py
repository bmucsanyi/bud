"""Direct risk prediction implementation as a wrapper class"""

import re

import torch
from torch import nn

from bud.utils import NonNegativeRegressor
from bud.wrappers.model_wrapper import SpecialWrapper


class BaseRiskPredictionWrapper(SpecialWrapper):
    pass


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels, width, pool):
        super().__init__()

        # Embedding layers
        self.norm = nn.LayerNorm(in_channels)
        self.pool = pool
        self.linear = nn.Linear(in_channels, width)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        use_norm = x.dim() < 4

        if use_norm:
            x = self.norm(x)

        x = self.pool(x)

        if use_norm:
            x = self.norm(x)

        x = self.linear(x)
        x = self.leaky_relu(x)

        return x


class DeepRiskPredictionWrapper(BaseRiskPredictionWrapper):
    """
    This module takes a model as input and creates a deep risk prediction module from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,  # 256
        mlp_depth: int,
        stopgrad: bool,
        num_hooks=None,
        module_type=None,
        module_name_regex=None,
    ):
        super().__init__(model)

        self.num_hidden_features = num_hidden_features
        self.mlp_depth = mlp_depth
        self.stopgrad = stopgrad
        self.num_hooks = num_hooks
        self.module_type = module_type
        self.module_name_regex = module_name_regex

        # Register hooks to extract intermediate features
        self.feature_buffer = {}
        self.hook_handles = []
        self.hook_layer_names = []

        layer_candidates = self.get_layer_candidates(
            model, module_type, module_name_regex
        )
        chosen_layers = self.filter_layer_candidates(layer_candidates, num_hooks)
        self.attach_hooks(chosen_layers)

        # Initialize uncertainty network(s)
        self.add_embedding_modules()
        self.regressor = NonNegativeRegressor(
            in_channels=num_hidden_features * num_hooks,
            width=num_hidden_features,
            depth=mlp_depth,
        )

    @staticmethod
    def get_layer_candidates(model, module_type, module_name_regex):
        layer_candidates = {}

        if module_name_regex is not None:
            module_name_regex = re.compile(module_name_regex)

        for name, module in model.named_modules():
            if (module_name_regex is not None and module_name_regex.match(name)) or (
                module_type is not None and isinstance(module, module_type)
            ):
                layer_candidates[name] = module

        return layer_candidates

    @staticmethod
    def filter_layer_candidates(layer_candidates, num_hooks):
        if num_hooks is None:
            return layer_candidates

        num_hooks = min(num_hooks, len(layer_candidates))

        chosen_layers = []
        chosen_indices = torch.linspace(
            start=0, end=len(layer_candidates) - 1, steps=num_hooks
        )
        chosen_layers = {
            name: module
            for i, (name, module) in enumerate(layer_candidates.items())
            if i in chosen_indices
        }

        return chosen_layers

    def attach_hooks(self, chosen_layers):
        def get_features(name):
            def hook(model, input, output):
                self.feature_buffer[name] = output.detach() if self.stopgrad else output

            return hook

        # Attach hooks to all children layers
        for name, layer in chosen_layers.items():
            self.hook_layer_names.append(name)
            handle = layer.register_forward_hook(get_features(name))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def add_embedding_modules(self):
        # Get the feature map sizes
        empty_image = torch.zeros(
            [1, *self.model.default_cfg["input_size"]],
            device=next(self.model.parameters()).device,
        )
        with torch.no_grad():
            self.feature_buffer.clear()
            self.model(empty_image)
            feature_sizes = {
                key: self.get_feature_dim(feature.shape)
                for key, feature in self.feature_buffer.items()
            }

        modules = {}
        self.hook_layer_name_to_embedding_module = {}
        for i, (key, size) in enumerate(feature_sizes.items()):
            module_name = f"unc_{i}"
            modules[module_name] = EmbeddingNetwork(
                size,
                self.num_hidden_features,
                pool=self.get_pooling_layer(self.feature_buffer[key].shape),
            )
            self.hook_layer_name_to_embedding_module[key] = module_name
        self.embedding_modules = nn.ModuleDict(modules)

    @staticmethod
    def get_feature_dim(shape):
        # Exclude the batch dimension
        dims = shape[1:]

        # If there's only one dimension, return nn.Identity
        if len(dims) == 1:
            return dims[0]
        elif len(dims) == 2:
            return dims[1]
        elif len(dims) == 3:
            return dims[0]
        else:
            raise ValueError("Invalid network structure.")

    @staticmethod
    def get_pooling_layer(shape):
        # Exclude the batch dimension
        dims = shape[1:]

        # If there's only one dimension, return nn.Identity
        if len(dims) == 1:
            return nn.Identity()
        elif len(dims) == 2:
            dim = 1
        elif len(dims) == 3:
            dim = (2, 3)
        else:
            raise ValueError("Invalid network structure.")

        return AveragePool(dim)

    def forward_features(self, x):
        self.feature_buffer.clear()

        return self.model.forward_features(x)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)
        regressor_features = torch.cat(
            [
                self.embedding_modules[
                    self.hook_layer_name_to_embedding_module[hook_layer_name]
                ](self.feature_buffer[hook_layer_name])
                for hook_layer_name in self.hook_layer_names
            ],
            dim=1,
        )

        risk_values = self.regressor(regressor_features).squeeze()

        if self.training:
            return logits, risk_values
        else:
            return {
                "logit": logits,
                "feature": features,
                "risk_value": risk_values,
            }


class AveragePool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs.mean(self.dim)


class RiskPredictionWrapper(BaseRiskPredictionWrapper):
    """
    This module takes a model as input and creates a risk prediction module from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,  # 256
        mlp_depth: int,
        stopgrad: bool,
    ):
        super().__init__(model)

        self.num_hidden_features = num_hidden_features
        self.mlp_depth = mlp_depth
        self.stopgrad = stopgrad

        self.regressor = NonNegativeRegressor(
            in_channels=model.num_features,
            width=num_hidden_features,
            depth=mlp_depth,
        )

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)

        regressor_features = features.detach() if self.stopgrad else features

        risk_values = self.regressor(regressor_features).squeeze()

        if self.training:
            return logits, risk_values
        else:
            return {
                "logit": logits,
                "feature": features,
                "risk_value": risk_values,
            }
