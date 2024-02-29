"""MCInfoNCE implementation as a wrapper class based on
https://github.com/mkirchhof/url"""

import torch
import torch.nn as nn

from bud.utils.model import NonNegativeRegressor
from bud.wrappers.model_wrapper import SpecialWrapper


class MCInfoNCEHead(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        num_hidden_features,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden_features = num_hidden_features
        self.register_buffer(
            "batch_kappa_scaler", torch.tensor(1.0, dtype=torch.float32)
        )

        self.kappa_predictor = NonNegativeRegressor(
            in_channels=self.num_features,
            width=self.num_hidden_features,
            depth=3,
            eps=1e-6,
        )

    def update_kappa_scaler(self, kappa_scaler):
        self.batch_kappa_scaler = torch.tensor(kappa_scaler, dtype=torch.float32)

    def forward(self, features):
        # Build vMFs from the features
        batch_kappas = self.batch_kappa_scaler * self.kappa_predictor(
            features
        )  # [B, 1]

        if self.training:
            return features, batch_kappas
        else:
            return {
                "feature": features,  # [B, D]
                "mcinfonce_inverse_kappa": 1 / batch_kappas.squeeze(),  # [B]
            }


class MCInfoNCEWrapper(SpecialWrapper):
    """
    This module takes a model as input and creates an MCInfoNCE model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,  # 512
        initial_average_kappa,
    ):
        super().__init__(model)

        self.num_hidden_features = num_hidden_features
        self.initial_average_kappa = initial_average_kappa

        self.classifier = MCInfoNCEHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_hidden_features=self.num_hidden_features,
        )

    def initialize_average_kappa(
        self, train_loader, amp_autocast, device, num_batches=10
    ):
        # Find out what uncertainty the model currently predicts on average
        prev_state = self.training
        self.eval()

        average_kappa = 0
        data_iter = iter(train_loader)

        with torch.no_grad():
            for _ in range(num_batches):
                input, _ = next(data_iter)

                if device:
                    input = input.to(device)

                with amp_autocast():
                    inference_dict = self(input)
                kappa = 1 / inference_dict["mcinfonce_inverse_kappa"]
                average_kappa += kappa.mean().detach().cpu().item() / num_batches

        self.classifier.update_kappa_scaler(self.initial_average_kappa / average_kappa)
        self.train(prev_state)

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        num_hidden_features: int = None,
        *args,
        **kwargs,
    ):
        if num_hidden_features is not None:
            self.num_hidden_features = num_hidden_features

        self.model.reset_classifier(*args, **kwargs)
        self.classifier = MCInfoNCEHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_hidden_features=self.num_hidden_features,
        )

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)  # [B, D]

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        return out
