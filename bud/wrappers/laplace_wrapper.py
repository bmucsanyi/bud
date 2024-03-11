from laplace import Laplace
from torch import nn

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
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.weight_path = weight_path
        self.laplace_model = None
        self.is_last_layer_laplace = is_last_layer_laplace
        self.pred_type = pred_type
        self.prior_optimization_method = prior_optimization_method
        self.hessian_structure = hessian_structure

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
        self.laplace_model.optimize_prior_precision(
            method=self.prior_optimization_method, val_loader=val_loader
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
            return {
                "logit": self.laplace_model.predictive_samples(
                    x=inputs, pred_type=self.pred_type, n_samples=self.num_mc_samples
                )
                .log()
                .clamp(min=1e-10)
                .permute(1, 0, 2),  # [B, S, C]
                "feature": self.model.forward_head(
                    self.model.forward_features(inputs), pre_logits=True
                ),
            }
