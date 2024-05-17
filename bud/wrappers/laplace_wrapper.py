from laplace import Laplace
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import time

from bud.utils.replace import replace
from bud.wrappers.model_wrapper import PosteriorWrapper


class NonInplaceReLU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs):
        return self.relu(inputs)


class LaplaceWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a Laplace-approximated model posterior
    from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        weight_path: str,
        is_last_layer_laplace: bool,
        pred_type: str,  # "glm", "nn"
        prior_optimization_method: str,  # "marglik", "CV"
        hessian_structure: str,  # "kron", "full"
        link_approx: str,  # "probit", "mc"
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.weight_path = weight_path
        self.laplace_model = None
        self.is_last_layer_laplace = is_last_layer_laplace
        self.pred_type = pred_type
        self.prior_optimization_method = prior_optimization_method
        self.hessian_structure = hessian_structure
        self.link_approx = link_approx

        self.load_model()

        if not is_last_layer_laplace:
            replace(
                model,
                "ReLU",
                NonInplaceReLU,
            )

    def perform_laplace_approximation(self, train_loader, val_loader):
        subset_of_weights = "last_layer" if self.is_last_layer_laplace else "all"
        self.laplace_model = Laplace(
            self.model,
            "classification",
            subset_of_weights=subset_of_weights,
            hessian_structure=self.hessian_structure,
        )
        self.laplace_model.fit(train_loader)

        if self.prior_optimization_method == "CV":
            self.optimize_prior_precision_cv(
                pred_type=self.pred_type,
                val_loader=val_loader,
                link_approx=self.link_approx,
            )
        else:
            self.laplace_model.optimize_prior_precision(
                method=self.prior_optimization_method,
                pred_type=self.pred_type,
                val_loader=val_loader,
                link_approx=self.link_approx,
            )

    def forward_head(self, *args, **kwargs):
        # Warning! This class requires extra care, as the predictive samples are
        # sampled end-to-end from a black-box package. One can't use the usual strategy
        # of "obtain features => obtain logits". Instead, one has to obtain features
        # with `forward_features` and the logits with `forward`.
        raise ValueError(f"forward_head cannot be called directly for {type(self)}")

    def forward(self, inputs):
        if self.laplace_model is None:
            raise ValueError("Model has to be Laplace-approximated first.")

        if self.training:
            return self.model(inputs)
        else:
            feature = self.model.forward_head(
                self.model.forward_features(inputs), pre_logits=True
            )

            return {
                "logit": self.laplace_model.predictive_samples(
                    x=inputs, pred_type=self.pred_type, n_samples=self.num_mc_samples
                )
                .log()
                .clamp(min=torch.finfo(feature.dtype).min)
                .permute(1, 0, 2),  # [B, S, C]
                "feature": feature,
            }

    @staticmethod
    def get_nll(out_dist, targets):
        return F.nll_loss(
            out_dist.log().clamp(min=torch.finfo(out_dist.dtype).min), targets
        )

    def optimize_prior_precision_cv(
        self,
        pred_type,
        val_loader,
        link_approx="probit",
        log_prior_prec_min=-4,
        log_prior_prec_max=4,
        grid_size=100,
        n_samples=100,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.laplace_model.prior_precision = self.gridsearch(
            interval=interval,
            val_loader=val_loader,
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=n_samples,
        )

        print(f"Optimized prior precision is {self.laplace_model.prior_precision}.")

    def gridsearch(
        self,
        interval,
        val_loader,
        pred_type,
        link_approx="probit",
        n_samples=100,
    ):
        results = list()
        prior_precs = list()
        for prior_prec in interval:
            print(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self.laplace_model.prior_precision = prior_prec
            try:
                out_dist, targets = self.validate(
                    val_loader=val_loader,
                    pred_type=pred_type,
                    link_approx=link_approx,
                    n_samples=n_samples,
                )
                result = self.get_nll(out_dist, targets).item()
            except RuntimeError as error:
                print(f"Caught an exception in validate: {error}")
                result = float("inf")
            print(f"Took {time.perf_counter() - start_time} seconds")
            results.append(result)
            prior_precs.append(prior_prec)
        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def validate(
        self, val_loader, pred_type="glm", link_approx="probit", n_samples=100
    ):
        self.laplace_model.model.eval()
        output_means = list()
        targets = list()
        for X, y in val_loader:
            X, y = X.to(self.laplace_model._device), y.to(self.laplace_model._device)
            out = self.laplace_model(
                X, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples
            )

            output_means.append(out)
            targets.append(y)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)
