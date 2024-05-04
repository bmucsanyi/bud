import torch
from pyro.distributions.transforms.radial import Radial
from torch import nn
from torch.distributions import MultivariateNormal
from bud.wrappers.model_wrapper import DirichletWrapper
from bud.utils import entropy


class NormalizingFlowDensity(nn.Module):
    def __init__(self, dim: int, flow_length: int):
        super(NormalizingFlowDensity, self).__init__()
        self.dim = dim
        self.flow_length = flow_length

        self.register_buffer("standard_mean", torch.zeros(dim))
        self.register_buffer("standard_cov", torch.eye(dim))

        self.transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        sum_log_jacobians = 0
        z = x
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians += transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        log_prob_z = MultivariateNormal(self.standard_mean, self.standard_cov).log_prob(
            z
        )
        log_prob_x = log_prob_z + sum_log_jacobians

        return log_prob_x


class PostNetWrapper(DirichletWrapper):
    def __init__(
        self,
        model: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        num_density_components: int,
    ):
        super().__init__(model)

        # TODO: come back to check if these are needed/useful
        self.register_buffer("sample_count_per_class", None)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_density_components = num_density_components
        self.num_classes = model.num_classes

        # Use wrapped model as a feature extractor
        self.model.reset_classifier(num_classes=latent_dim)

        self.batch_norm = nn.BatchNorm1d(num_features=latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.num_classes),
        )

        self.density_estimator = nn.ModuleList(
            [
                NormalizingFlowDensity(
                    dim=self.latent_dim, flow_length=self.num_density_components
                )
                for _ in range(self.num_classes)
            ]
        )

    def calculate_sample_counts(self, train_loader):
        device = next(self.model.parameters()).device
        self.sample_count_per_class = torch.zeros((self.num_classes))

        for _, targets in train_loader:
            targets = targets.cpu()
            self.sample_count_per_class.scatter_add_(
                0, targets, torch.ones_like(targets, dtype=torch.float)
            )

        self.sample_count_per_class.to(device)
        print(self.sample_count_per_class)

    def forward_head(self, x, pre_logits: bool = False):
        if self.sample_count_per_class is None:
            raise ValueError("call to `calculate_sample_counts` needed first")

        # Pre-logits are the outputs of the wrapped model
        features = self.batch_norm(self.model.forward_head(x))  # [B, D]

        if pre_logits:
            return features

        if isinstance(self.density_estimator, nn.ModuleList):
            batch_size = features.shape[0]
            log_probs = torch.zeros(
                (batch_size, self.num_classes), device=features.device
            )
            alphas = torch.zeros((batch_size, self.num_classes), device=features.device)

            for c in range(self.num_classes):
                log_probs[:, c] = self.density_estimator[c].log_prob(features)
                alphas[:, c] = (
                    1 + self.sample_count_per_class[c] * log_probs[:, c].exp()
                )
        else:
            # TODO: implement for imagenet, current one likely infeasible
            log_probs = self.density_estimator.log_prob(features)  # [C, B]
            alphas = (
                1 + self.sample_count_per_class.unsqueeze(1).mul(log_probs.exp()).T
            )  # [B, C]

        if self.training:
            return alphas

        sum_alphas = alphas.sum(dim=1)  # [B]

        num_classes = alphas.shape[1]
        dirichlet_scaled_inverse_precision = num_classes / sum_alphas  # [B]

        mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]
        digamma_term = torch.digamma(alphas + 1) - torch.digamma(
            sum_alphas + 1
        ).unsqueeze(
            1
        )  # [B, C]
        dirichlet_expected_entropy = -mean_alphas.mul(digamma_term).sum(dim=1)  # [B]

        dirichlet_entropy_of_expectation = entropy(mean_alphas)  # [B]

        dirichlet_mutual_information = (
            dirichlet_entropy_of_expectation - dirichlet_expected_entropy
        )

        return {
            "mean_alpha": mean_alphas,  # [B, C]
            "feature": features,  # [B, D]
            "dirichlet_scaled_inverse_precision": dirichlet_scaled_inverse_precision,  # [B]
            "dirichlet_expected_entropy": dirichlet_expected_entropy,  # [B]
            "dirichlet_entropy_of_expectation": dirichlet_entropy_of_expectation,  # [B]
            "dirichlet_mutual_information": dirichlet_mutual_information,  # [B]
        }

    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        hidden_dim: int = None,
        num_classes: int = None,
    ):
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim

        if num_classes is not None:
            self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes),
        )
