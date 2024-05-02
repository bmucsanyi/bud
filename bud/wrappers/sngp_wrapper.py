"""SNGP/GP implementation as a wrapper class.

SNGP implementation based on https://github.com/google/edward2
"""

from functools import partial
import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import is_lazy
from torch.nn.modules.batchnorm import _NormBase


from bud.utils.replace import register, register_cond, replace
from bud.wrappers.model_wrapper import PosteriorWrapper
from bud.utils import calculate_same_padding, calculate_output_padding


class GPOutputLayer(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_mc_samples,
        num_random_features,
        gp_kernel_scale,
        gp_output_bias,
        gp_random_feature_type,
        is_gp_input_normalized,
        gp_cov_momentum,
        gp_cov_ridge_penalty,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_random_features = num_random_features
        self.num_mc_samples = num_mc_samples

        self.is_gp_input_normalized = is_gp_input_normalized
        self.gp_input_scale = (
            1 / gp_kernel_scale**0.5 if gp_kernel_scale is not None else None
        )

        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias

        if gp_random_feature_type == "orf":
            self.random_features_weight_initializer = partial(
                self.orthogonal_random_features_initializer, std=0.05
            )
        elif gp_random_feature_type == "rff":
            self.random_features_weight_initializer = partial(
                nn.init.normal_, mean=0.0, std=0.05
            )
        else:
            raise ValueError(
                "gp_random_feature_type must be one of 'orf' or 'rff', got "
                f"{gp_random_feature_type}"
            )

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty

        # Default to Gaussian RBF kernel with orthogonal random features.
        self.random_features_bias_initializer = partial(
            nn.init.uniform_, a=0, b=2 * torch.pi
        )

        if self.is_gp_input_normalized:
            self.input_norm_layer = nn.LayerNorm(num_features)

        self.random_feature = self.make_random_feature_layer(num_features)

        self.gp_cov_layer = LaplaceRandomFeatureCovariance(
            gp_feature_dim=self.num_random_features,
            momentum=self.gp_cov_momentum,
            ridge_penalty=self.gp_cov_ridge_penalty,
        )

        self.gp_output_layer = nn.Linear(
            in_features=self.num_random_features,
            out_features=self.num_classes,
            bias=False,
        )

        self.gp_output_bias = nn.Parameter(
            torch.tensor([self.gp_output_bias] * self.num_classes), requires_grad=False
        )

    @staticmethod
    def orthogonal_random_features_initializer(tensor, std):
        num_rows, num_cols = tensor.shape
        if num_rows < num_cols:
            # When num_rows < num_cols, sample multiple (num_rows, num_rows) matrices and
            # then concatenate.
            ortho_mat_list = []
            num_cols_sampled = 0

            while num_cols_sampled < num_cols:
                matrix = torch.empty_like(tensor[:, :num_rows])
                ortho_mat_square = nn.init.orthogonal_(matrix, gain=std)
                ortho_mat_list.append(ortho_mat_square)
                num_cols_sampled += num_rows

            # Reshape the matrix to the target shape (num_rows, num_cols)
            ortho_mat = torch.cat(ortho_mat_list, dim=-1)
            ortho_mat = ortho_mat[:, :num_cols]
        else:
            matrix = torch.empty_like(tensor)
            ortho_mat = nn.init.orthogonal_(matrix, gain=std)

        # Sample random feature norms.
        # Construct Monte-Carlo estimate of squared column norm of a random
        # Gaussian matrix.
        feature_norms_square = torch.randn_like(ortho_mat) ** 2
        feature_norms = feature_norms_square.sum(dim=0).sqrt()

        # Sets a random feature matrix with orthogonal column and Gaussian-like
        # column norms.
        value = ortho_mat * feature_norms
        with torch.no_grad():
            tensor.copy_(value)

        return tensor

    def make_random_feature_layer(self, num_features):
        """Defines random feature layer depending on kernel type."""
        # Use user-supplied configurations.
        custom_random_feature_layer = nn.Linear(
            in_features=num_features,
            out_features=self.num_random_features,
        )
        self.random_features_weight_initializer(custom_random_feature_layer.weight)
        self.random_features_bias_initializer(custom_random_feature_layer.bias)
        custom_random_feature_layer.weight.requires_grad_(False)
        custom_random_feature_layer.bias.requires_grad_(False)

        return custom_random_feature_layer

    def reset_covariance_matrix(self):
        """Resets covariance matrix of the GP layer.

        This function is useful for reseting the model's covariance matrix at the
        begining of a new epoch.
        """
        self.gp_cov_layer.reset_precision_matrix()

    @staticmethod
    def mean_field_logits(logits, covmat, mean_field_factor):
        # Compute standard deviation.
        variances = covmat.diag()

        # Compute scaling coefficient for mean-field approximation.
        logits_scale = (1 + variances * mean_field_factor).sqrt()

        # Cast logits_scale to compatible dimension.
        logits_scale = logits_scale.reshape(-1, 1)

        return logits / logits_scale

    @staticmethod
    def monte_carlo_sample_logits(logits, covmat, num_samples):
        # logits: [B, C]
        # covmat: [B, B]
        batch_size, num_classes = logits.shape
        vars = covmat.diag().reshape(-1, 1, 1)  # [B, 1, 1]

        # std_normal_samples: [B, S, C]
        std_normal_samples = torch.randn(
            batch_size, num_samples, num_classes, device=logits.device
        )

        return vars.sqrt() * std_normal_samples + logits.unsqueeze(dim=1)

    def forward(self, gp_inputs):
        # Computes random features.
        if self.is_gp_input_normalized:
            gp_inputs = self.input_norm_layer(gp_inputs)
        elif self.gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly
            # rescaling the input.
            gp_inputs *= self.gp_input_scale

        gp_feature = self.random_feature(gp_inputs).cos()

        # Computes posterior center (i.e., MAP estimate) and variance
        gp_output = self.gp_output_layer(gp_feature) + self.gp_output_bias  # [B, C]

        if self.training:
            return gp_output  # [B, C]

        with torch.no_grad():
            gp_covmat = self.gp_cov_layer(gp_feature)

        logits = self.monte_carlo_sample_logits(
            gp_output, gp_covmat, self.num_mc_samples
        )

        # When one wants only the BMA, the mean-field approximation is a solid choice.
        # logits = self.mean_field_logits(gp_output, gp_covmat, mean_field_factor=1)
        return logits  # [B, S, C]


class LaplaceRandomFeatureCovariance(nn.Module):
    def __init__(
        self,
        gp_feature_dim,
        momentum,
        ridge_penalty,
    ):
        super().__init__()
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        # Posterior precision matrix for the GP's random feature coefficients
        precision_matrix = torch.zeros((gp_feature_dim, gp_feature_dim))
        self.register_buffer("precision_matrix", precision_matrix)
        covariance_matrix = torch.eye(gp_feature_dim)
        self.register_buffer("covariance_matrix", covariance_matrix)

        # Boolean flag to indicate whether to update the covariance matrix (i.e.,
        # by inverting the newly updated precision matrix) during inference.
        self.is_update_covariance = False

    def update_feature_precision_matrix(self, gp_feature):
        """Computes the update precision matrix of feature weights."""
        batch_size = gp_feature.shape[0]

        # Computes batch-specific normalized precision matrix.
        precision_matrix_minibatch = gp_feature.T @ gp_feature

        # Updates the population-wise precision matrix.
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.momentum * self.precision_matrix
                + (1 - self.momentum) * precision_matrix_minibatch
            )
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch

        return precision_matrix_new

    def reset_precision_matrix(self):
        """Resets precision matrix to its initial value.

        This function is useful for reseting the model's covariance matrix at the
        begining of a new epoch.
        """
        self.precision_matrix.zero_()

    def update_feature_covariance_matrix(self):
        """Computes the feature covariance if self.is_update_covariance=True.

        GP layer computes the covariancce matrix of the random feature coefficient
        by inverting the precision matrix. Since this inversion op is expensive,
        we will invoke it only when there is new update to the precision matrix
        (where self.is_update_covariance will be flipped to `True`.).

        Returns:
        The updated covariance_matrix.
        """
        precision_matrix = self.precision_matrix
        covariance_matrix = self.covariance_matrix
        gp_feature_dim = precision_matrix.shape[0]

        # Compute covariance matrix update only when `is_update_covariance = True`.
        if self.is_update_covariance:
            covariance_matrix_updated = torch.linalg.inv(
                self.ridge_penalty
                * torch.eye(gp_feature_dim, device=precision_matrix.device)
                + precision_matrix
            )
        else:
            covariance_matrix_updated = covariance_matrix

        return covariance_matrix_updated

    def compute_predictive_covariance(self, gp_feature):
        """Computes posterior predictive variance.

        Approximates the Gaussian process posterior using random features.
        Given training random feature Phi_tr (num_train, num_hidden) and testing
        random feature Phi_ts (batch_size, num_hidden). The predictive covariance
        matrix is computed as (assuming Gaussian likelihood):

        s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

        where s is the ridge factor to be used for stablizing the inverse, and I is
        the identity matrix with shape (num_hidden, num_hidden).

        Args:
        gp_feature: (torch.Tensor) The random feature of testing data to be used for
            computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
        (torch.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
        """
        # Computes the covariance matrix of the gp prediction.
        gp_cov_matrix = (
            self.ridge_penalty * gp_feature @ self.covariance_matrix @ gp_feature.T
        )

        return gp_cov_matrix

    def forward(self, inputs):
        """Minibatch updates the GP's posterior precision matrix estimate.

        Args:
        inputs: (torch.Tensor) GP random features, shape (batch_size,
            gp_hidden_size).
        logits: (torch.Tensor) Pre-activation output from the model. Needed
            for Laplace approximation under a non-Gaussian likelihood.
        training: (torch.bool) whether or not the layer is in training mode. If in
            training mode, the gp_weight covariance is updated using gp_feature.

        Returns:
        gp_stddev (torch.Tensor): GP posterior predictive variance,
            shape (batch_size, batch_size).
        """
        batch_size = inputs.shape[0]

        if self.training:
            # Computes the updated feature precision matrix.
            precision_matrix_updated = self.update_feature_precision_matrix(
                gp_feature=inputs
            )

            # Updates precision matrix.
            self.precision_matrix.copy_(precision_matrix_updated)

            # Enables covariance update in the next inference call.
            self.is_update_covariance = True

            # Return null estimate during training.
            return torch.eye(batch_size, device=inputs.device)
        else:
            # Lazily computes feature covariance matrix during inference.
            covariance_matrix_updated = self.update_feature_covariance_matrix()

            # Store updated covariance matrix.
            self.covariance_matrix.copy_(covariance_matrix_updated)

            # Disable covariance update in future inference calls (to avoid the
            # expensive torch.linalg.inv op) unless there are new update to precision
            # matrix.
            self.is_update_covariance = False

            return self.compute_predictive_covariance(gp_feature=inputs)


class LinearSpectralNormalizer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        dim: int,
        eps: float,
    ) -> None:
        super().__init__()

        weight = module.weight
        ndim = weight.ndim

        self.spectral_normalization_iteration = spectral_normalization_iteration
        self.spectral_normalization_bound = spectral_normalization_bound
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps

        if ndim > 1:
            weight_matrix = self._reshape_weight_to_matrix(weight)
            height, width = weight_matrix.shape

            u = weight_matrix.new_empty(height).normal_(mean=0, std=1)
            v = weight_matrix.new_empty(width).normal_(mean=0, std=1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            self._power_method(
                weight_matrix=weight_matrix, spectral_normalization_iteration=15
            )

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if self.dim > 0:
            # Permute self.dim to front
            weight = weight.permute(
                self.dim, *(dim for dim in range(weight.dim()) if dim != self.dim)
            )

        weight_matrix = weight.flatten(start_dim=1)

        return weight_matrix

    @torch.no_grad()
    def _power_method(
        self, weight_matrix: torch.Tensor, spectral_normalization_iteration: int
    ) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        assert weight_matrix.ndim == 2

        for _ in range(spectral_normalization_iteration):
            # u^\top W v = u^\top (\sigma u) = \sigma u^\top u = \sigma
            # where u and v are the first left and right (unit) singular vectors,
            # respectively. This power iteration produces approximations of u and v.
            self._u = F.normalize(
                weight_matrix @ self._v, dim=0, eps=self.eps, out=self._u
            )
            self._v = F.normalize(
                weight_matrix.T @ self._u, dim=0, eps=self.eps, out=self._v
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            weight_norm = weight.norm(p=2, dim=0).clamp_min(self.eps)
            division_factor = torch.max(
                torch.ones_like(weight_norm),
                weight_norm / self.spectral_normalization_bound,
            )

            return weight / division_factor
        else:
            weight_matrix = self._reshape_weight_to_matrix(weight=weight)

            if self.training:
                self._power_method(
                    weight_matrix=weight_matrix,
                    spectral_normalization_iteration=self.spectral_normalization_iteration,
                )

            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)

            sigma = u @ weight_matrix @ v
            division_factor = torch.max(
                torch.ones_like(sigma), sigma / self.spectral_normalization_bound
            )

            return weight / division_factor

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value

class Conv2dSpectralNormalizer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        eps: float,
    ) -> None:
        super().__init__()

        if (
            hasattr(module, "parametrizations")
            and hasattr(module.parametrizations, "weight")
            and Conv2dSpectralNormalizer in module.parametrizations.weight
        ):
            raise ValueError("Cannot register spectral normalization more than once.")

        weight = module.weight
        ndim = weight.ndim

        if ndim != 4:
            raise ValueError(
                f"Invalid weight shape: expected ndim = 4, received ndim = {ndim}"
            )

        self.spectral_normalization_iteration = spectral_normalization_iteration
        self.spectral_normalization_bound = spectral_normalization_bound
        self.eps = eps

        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.output_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.device = weight.device
        self.weight_shape = weight.shape

        # Partial initialization; shape inference happens in first forward
        self.input_shape = None

        self.register_buffer("_u", nn.UninitializedBuffer())
        self.register_buffer("_v", nn.UninitializedBuffer())

        self._load_hook = module._register_load_state_dict_pre_hook(
            self._lazy_load_hook
        )
        self._module_input_shape_hook = module.register_forward_pre_hook(
            Conv2dSpectralNormalizer._module_set_input_shape, with_kwargs=True
        )
        self._initialize_hook = self.register_forward_pre_hook(
            Conv2dSpectralNormalizer._infer_attributes, with_kwargs=True
        )

    @staticmethod
    def _module_set_input_shape(module, args, kwargs=None):
        kwargs = kwargs or {}
        input = kwargs["input"] if "input" in kwargs else args[0]

        for parametrization in module.parametrizations.weight:
            if isinstance(parametrization, Conv2dSpectralNormalizer):
                parametrization.input_shape = input.shape

                break  # Invariant: there is only one Conv2dSpectralNormalizer registered

    def _has_uninitialized_buffers(self) -> bool:
        buffers = self._buffers.values()
        for buffer in buffers:
            if is_lazy(buffer):
                return True
        return False

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # This should be ideally implemented as a hook,
        # but we should override `detach` in the UninitializedParameter to return itself
        # which is not clean
        for name, param in self._parameters.items():
            if param is not None:
                if not (is_lazy(param) or keep_vars):
                    param = param.detach()
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                if not (is_lazy(buf) or keep_vars):
                    buf = buf.detach()
                destination[prefix + name] = buf

    def _lazy_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        for name, param in itertools.chain(
            self._parameters.items(), self._buffers.items()
        ):
            key = prefix + name
            if key in state_dict and param is not None:
                input_param = state_dict[key]
                if is_lazy(param):
                    # The current parameter is not initialized but the one being loaded one is
                    # create a new parameter based on the uninitialized one
                    if not is_lazy(input_param):
                        with torch.no_grad():
                            param.materialize(input_param.shape)

    @staticmethod
    def _infer_attributes(module, args, kwargs=None):
        r"""Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferrable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
        # Infer buffers
        kwargs = kwargs or {}
        module._initialize_buffers(*args, **kwargs)
        if module._has_uninitialized_buffers():
            raise RuntimeError(
                f"module {module._get_name()} has not been fully initialized"
            )

        # Remove hooks
        module._load_hook.remove()
        module._module_input_shape_hook.remove()
        module._initialize_hook.remove()
        delattr(module, "_load_hook")
        delattr(module, "_module_input_shape_hook")
        delattr(module, "_initialize_hook")

    def _initialize_buffers(self, weight) -> None:
        if self._has_uninitialized_buffers():
            with torch.no_grad():
                # Infer input shape with batch size = 1
                # By now, the hook attached to the Conv2d module has computed the
                # input_shape attribute of self. Note that directly having the Conv2d
                # module as an attribute would lead to circular references and
                # RecursionErrors
                input_channels, input_height, input_width = self.input_shape[1:]
                self.single_input_shape = (1, input_channels, input_height, input_width)
                flattened_input_shape = math.prod(self.single_input_shape)

                # Infer output shape with batch size = 1. We know this without having
                # to run the Conv2d module, as we use "same" padding in our internal
                # calculations
                output_channels = self.output_channels
                output_height = input_height // self.stride[0]
                output_width = input_width // self.stride[1]
                self.single_output_shape = (
                    1,
                    output_channels,
                    output_height,
                    output_width,
                )
                flattened_output_shape = math.prod(self.single_output_shape)

                device = self.device

                # Materialize buffers
                self._u.materialize(shape=flattened_output_shape, device=device)
                self._v.materialize(shape=flattened_input_shape, device=device)

                # Initialize buffers randomly
                nn.init.normal_(self._u)
                nn.init.normal_(self._v)

                # Infer input padding
                self.left_right_top_bottom_padding = calculate_same_padding(
                        self.single_output_shape,
                        self.weight_shape,
                        self.stride,
                        self.dilation,
                    )
                total_width_height_padding = (
                    self.left_right_top_bottom_padding[0]
                    + self.left_right_top_bottom_padding[1],
                    self.left_right_top_bottom_padding[2]
                    + self.left_right_top_bottom_padding[3],
                )
                self.per_side_width_height_padding = (
                    math.ceil(total_width_height_padding[0] / 2),
                    math.ceil(total_width_height_padding[1] / 2),
                )

                # Infer output padding
                self.output_padding = calculate_output_padding(
                    input_shape=self.single_output_shape,
                    output_shape=self.single_input_shape,
                    stride=self.stride,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                )

                # Initialize buffers with correct values. We do 50 iterations to have
                # a good approximation of the correct singular vectors and the value
                self._power_method(weight=weight, spectral_normalization_iteration=50)

    def _replicate_for_data_parallel(self):
        if self._has_uninitialized_buffers():
            raise RuntimeError(
                "Modules with uninitialized parameters can't be used with `DataParallel`. "
                "Run a dummy forward pass to correctly initialize the modules"
            )

        return super()._replicate_for_data_parallel()

    # TODO: add dummy fw pass if data parallel needed; add it in later release

    @torch.no_grad()
    def _power_method(
        self, weight: torch.Tensor, spectral_normalization_iteration: int
    ) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        for _ in range(spectral_normalization_iteration):
            # u^\top W v = u^\top (\sigma u) = \sigma u^\top u = \sigma
            # where u and v are the first left and right (unit) singular vectors,
            # respectively. This power iteration produces approximations of u and v.

            # TODO: possibly get rid of "same" padding?
            v_shaped = F.conv_transpose2d(
                input=self._u.view(self.single_output_shape),
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.per_side_width_height_padding,
                output_padding=self.output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )

            self._v = F.normalize(
                input=v_shaped.view(-1), dim=0, eps=self.eps, out=self._v
            )

            v_padded = F.pad(
                self._v.view(self.single_input_shape),
                self.left_right_top_bottom_padding,
            )
            u_shaped = F.conv2d(
                input=v_padded,
                weight=weight,
                bias=None,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            )

            self._u = F.normalize(
                input=u_shaped.view(-1), dim=0, eps=self.eps, out=self._u
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(
                weight=weight,
                spectral_normalization_iteration=self.spectral_normalization_iteration,
            )

        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        # Pad v to have the "same" padding effect
        v_padded = F.pad(
            v.view(self.single_input_shape), self.left_right_top_bottom_padding
        )

        # Apply the _unnormalized_ weight to v
        weight_v = F.conv2d(
            input=v_padded,
            weight=weight,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Estimate largest singular value
        sigma = weight_v.view(-1) @ u.view(-1)

        # Calculate factor to divide weight by; pay attention to numerical stability
        division_factor = torch.max(
            torch.ones_like(sigma), sigma / self.spectral_normalization_bound
        ).clamp_min(self.eps)

        return weight / division_factor

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _SpectralNormalizedBatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        spectral_normalization_bound: float,
        eps: float = 1e-5,
        momentum: float = 0.01,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # Momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power
        # iteration. Code is based on torch.nn.modules._NormBase

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.spectral_normalization_bound = spectral_normalization_bound

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # If statement only here to tell the jit to skip emitting this when it
            # is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather
        # than the buffers. Mini-batch stats are used in training mode, and in eval mode
        # when buffers are None
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training
        # mode. Thus they only need to be passed when the update should occur (i.e. in
        # training mode when they are tracked), or when buffer stats are used for
        # normalization (i.e. in eval mode when buffers are not None)

        # Before the foward pass, estimate the Lipschitz constant of the layer and
        # divide through by it so that the Lipschitz constant of the batch norm operator
        # is at most self.coeff
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )
        # See https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct
        lipschitz = torch.max(torch.abs(weight * (self.running_var + self.eps) ** -0.5))

        # If the Lipschitz constant of the operation is greater than coeff, then we want
        # to divide the input by a constant to force the overall Lipchitz factor of the
        # batch norm to be exactly coeff
        lipschitz_factor = torch.max(
            lipschitz / self.spectral_normalization_bound, torch.ones_like(lipschitz)
        )

        weight = weight / lipschitz_factor

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SpectralNormalizedBatchNorm2d(_SpectralNormalizedBatchNorm):
    def __init__(self, module, spectral_normalization_bound: float) -> None:
        # TODO: set bn-momentum to 0.01 if we use this!
        super().__init__(
            module.num_features,
            spectral_normalization_bound,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class SNGPWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates an SNGP from it.
    """

    def __init__(
        self,
        model: nn.Module,
        is_spectral_normalized: bool,
        use_tight_norm_for_pointwise_convs: bool,  # TODO: read up on the paper again
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        is_batch_norm_spectral_normalized: bool,
        num_mc_samples: int,
        num_random_features: int,
        gp_kernel_scale: float,
        gp_output_bias: float,
        gp_random_feature_type: str,
        is_gp_input_normalized: bool,
        gp_cov_momentum: float,
        gp_cov_ridge_penalty: float,
        gp_input_dim: int,
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.num_random_features = num_random_features
        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias
        self.gp_random_feature_type = gp_random_feature_type
        self.is_gp_input_normalized = is_gp_input_normalized
        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_input_dim = gp_input_dim

        classifier = nn.Sequential()

        if self.gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self.gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self.gp_input_dim

            classifier.append(random_projection)
        else:
            num_gp_features = self.num_features

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self.num_mc_samples,
            num_random_features=self.num_random_features,
            gp_kernel_scale=self.gp_kernel_scale,
            gp_output_bias=self.gp_output_bias,
            gp_random_feature_type=self.gp_random_feature_type,
            is_gp_input_normalized=self.is_gp_input_normalized,
            gp_cov_momentum=self.gp_cov_momentum,
            gp_cov_ridge_penalty=self.gp_cov_ridge_penalty,
        )
        classifier.append(gp_output_layer)

        self.classifier = classifier

        if is_spectral_normalized:
            LSN = partial(
                LinearSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                dim=0,
                eps=1e-12,
            )

            CSN = partial(
                Conv2dSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                eps=1e-12,
            )

            SNBN = partial(
                SpectralNormalizedBatchNorm2d,
                spectral_normalization_bound=spectral_normalization_bound,
            )

            if use_tight_norm_for_pointwise_convs:

                def is_pointwise_conv(conv2d: nn.Conv2d) -> bool:
                    return conv2d.kernel_size == (1, 1)

                register_cond(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    cond=is_pointwise_conv,
                    target_parametrization_true=LSN,
                    target_parametrization_false=CSN,
                )
            else:
                register(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    target_parametrization=CSN,
                )

            if is_batch_norm_spectral_normalized:
                replace(
                    model=model,
                    source_regex="BatchNorm2d",
                    target_module=SNBN,
                )

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        num_mc_samples: int = None,
        num_random_features: int = None,
        gp_kernel_scale: float = None,
        gp_output_bias: float = None,
        gp_random_feature_type: str = None,
        is_gp_input_normalized: bool = None,
        gp_cov_momentum: float = None,
        gp_cov_ridge_penalty: float = None,
        gp_input_dim: int = None,
        *args,
        **kwargs,
    ):
        if num_mc_samples is not None:
            self.num_mc_samples = num_mc_samples

        if num_random_features is not None:
            self.num_random_features = num_random_features

        if gp_kernel_scale is not None:
            self.gp_kernel_scale = gp_kernel_scale

        if gp_output_bias is not None:
            self.gp_output_bias = gp_output_bias

        if gp_random_feature_type is not None:
            self.gp_random_feature_type = gp_random_feature_type

        if is_gp_input_normalized is not None:
            self.is_gp_input_normalized = is_gp_input_normalized

        if gp_cov_momentum is not None:
            self.gp_cov_momentum = gp_cov_momentum

        if gp_cov_ridge_penalty is not None:
            self.gp_cov_ridge_penalty = gp_cov_ridge_penalty

        if gp_input_dim is not None:
            self.gp_input_dim = gp_input_dim

        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        classifier = nn.Sequential()

        if self.gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self.gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self.gp_input_dim

            classifier.append(random_projection)
        else:
            num_gp_features = self.num_features

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self.num_mc_samples,
            num_random_features=self.num_random_features,
            gp_kernel_scale=self.gp_kernel_scale,
            gp_output_bias=self.gp_output_bias,
            gp_random_feature_type=self.gp_random_feature_type,
            is_gp_input_normalized=self.is_gp_input_normalized,
            gp_cov_momentum=self.gp_cov_momentum,
            gp_cov_ridge_penalty=self.gp_cov_ridge_penalty,
        )
        classifier.append(gp_output_layer)

        self.classifier = classifier
