"""PostNet (re)implementation based on https://github.com/sharpenb/Posterior-Network."""
import torch
from pyro.distributions.transforms.radial import Radial
from torch import nn
from torch.distributions import MultivariateNormal
from bud.wrappers.model_wrapper import DirichletWrapper
import math
from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import Transform, constraints
import torch.distributions as tdist
import torch.nn.functional as F


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


@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        raise NotImplementedError()
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = (
            self._params() if callable(self._params) else self._params
        )

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = -(h**2)
        beta_h = beta[:, None] * h

        self._cached_logDetJ = (x0.size(-1) - 1) * torch.log1p(beta_h) + torch.log1p(
            beta_h + beta[:, None] * h_prime * r
        )
        return x + beta_h[:, :, None] * diff

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError(
            "ConditionedRadial object expected to find key in intermediates cache but didn't"
        )

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedRadial)
class BatchedRadial(ConditionedRadial, TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c, input_dim):
        super().__init__(self._params)

        self.x0 = nn.Parameter(
            torch.Tensor(
                c,
                input_dim,
            )
        )
        self.alpha_prime = nn.Parameter(
            torch.Tensor(
                c,
            )
        )
        self.beta_prime = nn.Parameter(
            torch.Tensor(
                c,
            )
        )
        self.c = c
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.x0, self.alpha_prime, self.beta_prime

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.x0.size(1))
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)


class BatchedNormalizingFlowDensity(nn.Module):
    def __init__(self, c, dim, flow_length):
        super(BatchedNormalizingFlowDensity, self).__init__()
        self.c = c
        self.dim = dim
        self.flow_length = flow_length

        self.mean = nn.Parameter(torch.zeros(self.c, self.dim), requires_grad=False)
        self.cov = nn.Parameter(
            torch.eye(self.dim).repeat(self.c, 1, 1), requires_grad=False
        )

        self.transforms = nn.Sequential(
            *(BatchedRadial(c, dim) for _ in range(flow_length))
        )

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(
                z, z_next
            )
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(
            self.mean.repeat(z.size(1), 1, 1).permute(1, 0, 2),
            self.cov.repeat(z.size(1), 1, 1, 1).permute(1, 0, 2, 3),
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x


class PostNetWrapper(DirichletWrapper):
    def __init__(
        self,
        model: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        num_density_components: int,
        is_batched: bool,
    ):
        super().__init__(model)

        self.latent_dim = latent_dim
        self.num_features = latent_dim  # For compatibility
        self.hidden_dim = hidden_dim
        self.num_density_components = num_density_components
        self.num_classes = model.num_classes
        self.register_buffer("sample_count_per_class", torch.zeros(self.num_classes))

        # Use wrapped model as a feature extractor
        self.model.reset_classifier(num_classes=latent_dim)

        self.batch_norm = nn.BatchNorm1d(num_features=latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.num_classes),
        )

        if is_batched:
            self.density_estimator = BatchedNormalizingFlowDensity(
                c=self.num_classes,
                dim=self.latent_dim,
                flow_length=self.num_density_components,
            )
        else:
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
        sample_count_per_class = torch.zeros(self.num_classes)

        for _, targets in train_loader:
            targets = targets.cpu()
            sample_count_per_class.scatter_add_(
                0, targets, torch.ones_like(targets, dtype=torch.float)
            )

        self.sample_count_per_class = sample_count_per_class.to(device)

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
            log_probs = self.density_estimator.log_prob(features)  # [C, B]
            alphas = (
                1 + self.sample_count_per_class.unsqueeze(1).mul(log_probs.exp()).T
            )  # [B, C]

        if self.training:
            return alphas

        return {
            "alpha": alphas,  # [B, C]
            "feature": features,  # [B, D]
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
