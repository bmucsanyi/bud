"""Mahalanobis implementation as a wrapper class. Latent density estimation based on
https://github.com/pokaxpoka/deep_Mahalanobis_detector"""

import re

import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch import nn

from bud.wrappers.model_wrapper import SpecialWrapper


class MahalanobisWrapper(SpecialWrapper):
    """
    This module takes a model as input and creates a Mahalanobis model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        magnitude: float,
        weight_path: str,
        num_hooks=None,
        module_type=None,
        module_name_regex=None,
    ):
        super().__init__(model)

        self.magnitude = magnitude
        self.weight_path = weight_path
        self.num_hooks = num_hooks
        self.module_type = module_type
        self.module_name_regex = module_name_regex

        # Register hooks to extract intermediate features
        self.feature_list = []
        self.hook_handles = []
        self.hook_layer_names = []
        self.logistic_regressor = None
        self.register_buffer("logistic_regressor_coef", None)
        self.register_buffer("logistic_regressor_intercept", None)

        layer_candidates = self.get_layer_candidates(
            model=self.model,
            module_type=self.module_type,
            module_name_regex=self.module_name_regex,
        )
        chosen_layers = self.filter_layer_candidates(
            layer_candidates=layer_candidates, num_hooks=self.num_hooks
        )
        self.num_layers = len(chosen_layers)
        self.attach_hooks(chosen_layers=chosen_layers)

        for i in range(self.num_layers):
            self.register_buffer(f"class_means_{i}", None)
            self.register_buffer(f"precisions_{i}", None)

        self.load_model()

    @staticmethod
    def pool_feature(feature):
        shape = feature.shape[1:]
        if len(shape) == 1:
            return feature
        elif len(shape) == 2:
            return feature.mean(dim=1)  # collapse dimension 1
        elif len(shape) == 3:
            return feature.mean(dim=(2, 3))
        else:
            raise ValueError("Invalid network structure.")

    def attach_hooks(self, chosen_layers):
        def hook(model, input, output):
            self.feature_list.append(self.pool_feature(output))

        # Attach hooks to all children layers
        for layer in chosen_layers:
            handle = layer.register_forward_hook(hook)
            self.hook_handles.append(handle)

    def remove_hooks(self):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def forward_head(self, x, pre_logits: bool = False):
        raise ValueError("Head cannot be called separately.")

    def forward(self, inputs):
        # This is the only way to interact with the model.
        if self.logistic_regressor is None:
            self.reconstruct_logistic_regressor()

        self.feature_list.clear()
        ret_features = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )
        ret_logits = self.model.get_classifier()(ret_features)

        noisy_mahalanobis_scores = self.calculate_noisy_mahalanobis_scores(
            inputs
        )  # [B, L]

        ret_uncertainties = torch.from_numpy(
            self.logistic_regressor.predict_proba(noisy_mahalanobis_scores.numpy())[
                :, 1
            ]
        )  # [B]

        return {
            "logit": ret_logits,
            "feature": ret_features,
            "mahalanobis_value": ret_uncertainties,
        }

    def calculate_noisy_mahalanobis_scores(self, inputs):
        noisy_mahalanobis_scores = torch.empty(0).to(
            next(self.model.parameters()).device
        )

        for layer_idx in range(self.num_layers):
            torch.set_grad_enabled(mode=True)
            inputs.requires_grad_(True)
            self.feature_list.clear()
            self.model(inputs)

            gradients = self.compute_gradients(
                inputs=inputs,
                features=self.feature_list[layer_idx],
                num_classes=self.model.num_classes,
                class_means=getattr(self, f"class_means_{layer_idx}"),
                precision_matrix=getattr(self, f"precisions_{layer_idx}"),
            )

            inputs.grad.zero_()
            torch.set_grad_enabled(mode=False)
            inputs.requires_grad_(False)

            temp_inputs = inputs - self.magnitude * gradients
            # Populate feature_list
            self.feature_list.clear()
            self.model(temp_inputs)
            noisy_mahalanobis_scores_layer = self.compute_gaussian_scores(
                features=self.feature_list[layer_idx],
                num_classes=self.model.num_classes,
                class_means=getattr(self, f"class_means_{layer_idx}"),
                precision_matrix=getattr(self, f"precisions_{layer_idx}"),
            ).max(dim=1, keepdim=True)[
                0
            ]  # [B, 1]

            noisy_mahalanobis_scores = torch.cat(
                [noisy_mahalanobis_scores, noisy_mahalanobis_scores_layer], dim=1
            )  # [B, L]

        return noisy_mahalanobis_scores.cpu()  # [B, L]

    def calculate_noisy_mahalanobis_scores_dataloader(
        self, dataloader, max_num_samples
    ):
        """
        Compute Mahalanobis confidence score on input dataloader.
        """
        mahalanobis_scores = torch.empty(0)

        num_samples = 0
        device = next(self.model.parameters()).device
        for inputs, _ in dataloader:
            if num_samples + inputs.shape[0] > max_num_samples:
                overhead = num_samples + inputs.shape[0] - max_num_samples
                modified_batch_size = inputs.shape[0] - overhead
                inputs = inputs[:modified_batch_size]

            inputs = inputs.to(device)
            noisy_mahalanobis_scores = self.calculate_noisy_mahalanobis_scores(
                inputs
            )  # [B, L]

            mahalanobis_scores = torch.cat(
                [mahalanobis_scores, noisy_mahalanobis_scores], dim=0
            )  # [N, L]

            num_samples += inputs.shape[0]
            if num_samples > max_num_samples:
                break

        return mahalanobis_scores  # [N, L]

    @staticmethod
    def get_layer_candidates(model, module_type, module_name_regex):
        layer_candidates = []

        if module_name_regex is not None:
            module_name_regex = re.compile(module_name_regex)

        for name, module in model.named_modules():
            if (module_name_regex is not None and module_name_regex.match(name)) or (
                module_type is not None and isinstance(module, module_type)
            ):
                layer_candidates.append(module)

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
        chosen_layers = [
            module for i, module in enumerate(layer_candidates) if i in chosen_indices
        ]

        return chosen_layers

    def train_logistic_regressor(
        self,
        train_loader,
        id_loader,
        ood_loader,
        max_num_training_samples=None,
        max_num_id_ood_samples=None,
    ):
        self.calculate_gaussian_parameters(
            train_loader=train_loader, max_num_training_samples=max_num_training_samples
        )

        num_id_samples = len(id_loader.dataset)
        num_ood_samples = len(ood_loader.dataset)

        if max_num_id_ood_samples is None:
            max_num_id_ood_samples = min(num_id_samples, num_ood_samples)
        else:
            max_num_id_ood_samples = min(
                min(num_id_samples, num_ood_samples), max_num_id_ood_samples
            )

        # Get Mahalanobis scores for in-distribution data
        mahalanobis_scores_id = self.calculate_noisy_mahalanobis_scores_dataloader(
            dataloader=id_loader, max_num_samples=max_num_id_ood_samples
        )
        labels_id = torch.zeros(mahalanobis_scores_id.shape[0])

        # Get Mahalanobis scores for out-of-distribution data
        mahalanobis_scores_ood = self.calculate_noisy_mahalanobis_scores_dataloader(
            dataloader=ood_loader, max_num_samples=max_num_id_ood_samples
        )
        labels_ood = torch.ones(mahalanobis_scores_ood.shape[0])

        # Concatenate scores and labels
        X_train = torch.cat(
            [mahalanobis_scores_id, mahalanobis_scores_ood], dim=0
        ).numpy()
        y_train = torch.cat([labels_id, labels_ood], dim=0).numpy()

        # Train logistic regression model
        logistic_regressor = LogisticRegressionCV(n_jobs=-1).fit(X_train, y_train)

        self.logistic_regressor = logistic_regressor
        self.logistic_regressor_coef = torch.from_numpy(self.logistic_regressor.coef_)
        self.logistic_regressor_intercept = torch.from_numpy(
            self.logistic_regressor.intercept_
        )

    def reconstruct_logistic_regressor(self):
        if (
            self.logistic_regressor_coef is None
            or self.logistic_regressor_intercept is None
        ):
            raise ValueError(
                "Logistic regressor weights are not set, nothing to reconstruct."
            )

        self.logistic_regressor = LogisticRegression()
        self.logistic_regressor.coef_ = self.logistic_regressor_coef.numpy()
        self.logistic_regressor.intercept_ = self.logistic_regressor_intercept.numpy()

    def calculate_gaussian_parameters(self, train_loader, max_num_training_samples):
        if max_num_training_samples is None:
            max_num_training_samples = len(train_loader.dataset)

        num_classes = self.model.num_classes
        num_layers = self.num_layers
        # Initialize tensors for storing features
        features_per_class_per_layer = [
            [[] for _ in range(num_classes)] for _ in range(num_layers)
        ]  # [L, C, *]

        # Process each batch
        num_training_samples = 0
        device = next(self.model.parameters()).device
        for inputs, targets in train_loader:
            # Truncate last batch to have exactly the maximum number of training samples
            if num_training_samples + inputs.shape[0] > max_num_training_samples:
                overhead = (
                    num_training_samples + inputs.shape[0] - max_num_training_samples
                )
                modified_batch_size = inputs.shape[0] - overhead
                inputs = inputs[:modified_batch_size]
                targets = targets[:modified_batch_size]

            inputs = inputs.to(device)
            self.feature_list.clear()
            self.model(inputs)
            feature_list = self.feature_list

            # Process each layer's output
            for layer_idx, feature in enumerate(feature_list):
                feature = feature.detach().cpu()
                for class_idx in range(num_classes):
                    class_features = feature[
                        targets.cpu() == class_idx
                    ]  # [N_{LC}, D_L]
                    if class_features.shape[0] > 0:
                        features_per_class_per_layer[layer_idx][class_idx].append(
                            class_features
                        )

            num_training_samples += inputs.shape[0]
            if num_training_samples >= max_num_training_samples:
                break

        # Aggregate and compute means and precision
        class_means = []  # [L, C, D_L]
        precisions = []  # [L, D_L, D_L]
        for i in range(num_layers):
            num_features = features_per_class_per_layer[i][0][0].shape[-1]
            class_means.append(
                torch.empty(
                    (num_classes, num_features),
                    device=device,
                )
            )
            precisions.append(torch.empty((num_features, num_features), device=device))

        for layer_idx in range(num_layers):
            layer_features = torch.empty(0)
            for class_idx in range(num_classes):
                # Concatenate all features for this class across batches
                class_features = torch.cat(
                    features_per_class_per_layer[layer_idx][class_idx], dim=0
                )  # [N_L, D_L]
                class_mean = class_features.mean(dim=0)  # [D_L]
                class_means[layer_idx][class_idx] = class_mean.to(device)

                centered_class_features = class_features - class_mean  # [N_L, D_L]

                # Aggregate features for precision calculation
                layer_features = torch.cat(
                    [layer_features, centered_class_features], dim=0
                )

            # Compute precision
            covariance = layer_features.T @ layer_features / layer_features.shape[0]
            precision = torch.linalg.pinv(covariance, hermitian=True)
            precisions[layer_idx] = precision.to(device)

        for i in range(self.num_layers):
            setattr(self, f"class_means_{i}", class_means[i])
            setattr(self, f"precisions_{i}", precisions[i])

    @staticmethod
    def compute_gaussian_scores(features, num_classes, class_means, precision_matrix):
        scores = []
        for class_idx in range(num_classes):
            difference = features.detach() - class_means[class_idx]  # [B, D_L]
            term = -0.5 * (difference @ precision_matrix @ difference.T).diag()  # [B]
            scores.append(term.unsqueeze(dim=1))  # [B, 1]

        return torch.cat(scores, dim=1)  # [B, C]

    @staticmethod
    def compute_gradients(inputs, features, num_classes, class_means, precision_matrix):
        gaussian_scores = MahalanobisWrapper.compute_gaussian_scores(
            features, num_classes, class_means, precision_matrix
        )

        max_score_indices = gaussian_scores.max(dim=1)[1]  # [B]
        max_means = class_means.index_select(dim=0, index=max_score_indices)  # [B, D_L]
        difference = features - max_means  # [B, D_L]
        term = -0.5 * (difference @ precision_matrix @ difference.T).diag()  # [B]
        loss = -term.mean()  # []
        loss.backward()

        gradients = inputs.grad.clone().sign()  # [B, C, H, W]

        return gradients
