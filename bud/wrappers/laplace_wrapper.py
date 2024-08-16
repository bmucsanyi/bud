from laplace import Laplace
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from torch.nn.utils import vector_to_parameters
from torch.distributions import MultivariateNormal

from bud.utils.metrics import calibration_error
from bud.wrappers.model_wrapper import PosteriorWrapper

logger = logging.getLogger(__name__)


class LaplaceWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a Laplace-approximated model posterior
    from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        num_mc_samples_cv: int,
        weight_path: str,
        pred_type: str,  # "glm", "nn"
        hessian_structure: str,  # "kron", "full", "diag"
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.num_mc_samples_cv = num_mc_samples_cv
        self.weight_path = weight_path
        self.laplace_model = None
        self.pred_type = pred_type
        self.hessian_structure = hessian_structure

        self.load_model()

    def perform_laplace_approximation(self, train_loader, val_loader):
        self.laplace_model = Laplace(
            self.model,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure=self.hessian_structure,
        )
        logger.info("Starting Laplace approximation.")
        self.laplace_model.fit(train_loader)
        logger.info("Laplace approximation done.")

        logger.info("Starting prior precision optimization.")
        self.optimize_prior_precision_cv(
            val_loader=val_loader,
        )
        logger.info("Prior precision optimization done.")

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
                "logit": self.logit_samples(
                    x=inputs,
                    num_samples=self.num_mc_samples,
                ),  # [B, S, C]
                "feature": feature,
            }

    @staticmethod
    def get_ece(out_dist, targets):
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def optimize_prior_precision_cv(
        self,
        val_loader,
        log_prior_prec_min=-1,
        log_prior_prec_max=3,
        grid_size=100,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.laplace_model.prior_precision = self.gridsearch(
            interval=interval,
            val_loader=val_loader,
        )

        logger.info(
            f"Optimized prior precision is {self.laplace_model.prior_precision}."
        )

    def gridsearch(
        self,
        interval,
        val_loader,
    ):
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self.laplace_model.prior_precision = prior_prec

            try:
                out_dist, targets = self.validate(
                    val_loader=val_loader,
                )
                result = self.get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy {accuracy}"
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def validate(self, val_loader):
        self.laplace_model.model.eval()
        output_means = []
        targets = []

        for X, y in val_loader:
            X, y = X.to(self.laplace_model._device), y.to(self.laplace_model._device)
            out = self.logit_samples(
                x=X,
                num_samples=self.num_mc_samples_cv,
            )  # [B, S, C]
            out = F.log_softmax(out, dim=-1).exp().mean(dim=1)  # [B, C]

            output_means.append(out)
            targets.append(y)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def nn_logit_samples(self, X, num_samples=100):
        fs = []

        for sample in self.laplace_model.sample(num_samples):
            vector_to_parameters(
                sample, self.laplace_model.model.last_layer.parameters()
            )
            fs.append(
                self.laplace_model.model(X.to(self.laplace_model._device)).detach()
            )

        vector_to_parameters(
            self.laplace_model.mean, self.laplace_model.model.last_layer.parameters()
        )
        fs = torch.stack(fs)

        return fs.permute(1, 0, 2)

    def glm_logit_distribution(self, X):
        Js, f_mu = self.laplace_model.backend.last_layer_jacobians(X)
        f_var = self.laplace_model.functional_variance(Js)

        return f_mu.detach(), f_var.detach()

    def logit_samples(self, x, num_samples=100):
        """Sample from the posterior logits on input data `x`.
        Can be used, for example, for Thompson sampling.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        num_samples : int
            number of samples

        Returns
        -------
        samples : torch.Tensor
            samples `(batch_size, num_samples, output_shape)`
        """
        if self.pred_type not in ["glm", "nn"]:
            raise ValueError("Only glm and nn supported as prediction types.")

        if self.pred_type == "glm":
            f_mu, f_var = self.glm_logit_distribution(x)
            dist = MultivariateNormal(f_mu, f_var)
            samples = dist.sample((num_samples,))

            return samples.permute(1, 0, 2)
        else:  # 'nn'
            return self.nn_logit_samples(x, num_samples)
