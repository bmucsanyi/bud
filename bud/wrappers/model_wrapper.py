"""
Contains base wrapper classes
"""

import torch
from torch import nn


class ModelWrapper(nn.Module):
    """General model wrapper base class."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def set_grads(
        self,
        backbone_requires_grad: bool = True,
        classifier_requires_grad: bool = True,
        wrapper_requires_grad: bool = False,
    ):
        # Freeze / unfreeeze parts of the model:
        # backbone, classifier head and bias predictor head
        params_classifier = [p for p in self.model.get_classifier().parameters()]
        params_classifier_plus_backbone = [p for p in self.model.parameters()]
        params_classifier_plus_backbone_plus_wrapper = [p for p in self.parameters()]

        for param in params_classifier_plus_backbone_plus_wrapper:
            param.requires_grad = wrapper_requires_grad
        for param in params_classifier_plus_backbone:
            param.requires_grad = backbone_requires_grad
        for param in params_classifier:
            param.requires_grad = classifier_requires_grad

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        return getattr(self.model, name)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        if self.training:
            return out
        else:
            return {"logit": out, "feature": features}

    @staticmethod
    def convert_state_dict(state_dict):
        """
        Convert state_dict by removing 'model.' prefix from keys.
        """
        converted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                converted_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                converted_state_dict[k] = v
        return converted_state_dict

    def load_model(self):
        """
        Load the model.
        """
        weight_path = self.weight_path
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        state_dict = self.convert_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


class PosteriorWrapper(ModelWrapper):
    pass


class SpecialWrapper(ModelWrapper):
    pass


class DirichletWrapper(PosteriorWrapper):
    pass
