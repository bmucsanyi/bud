"""SNGP/GP implementation as a wrapper class.

SNGP Google implementation based on https://github.com/google/edward2

SNGP Original implementation based on https://github.com/pfnet-research/sngan_projection

SNGP DUE implementation based on https://github.com/y0ast/DUE

"""

from functools import partial
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from bud.utils.replace import replace
from bud.wrappers.model_wrapper import PosteriorWrapper


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
        gp_stddev (tf.Tensor): GP posterior predictive variance,
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


class SpectralNormalizedConv2dGoogle(nn.Module):
    def __init__(
        self,
        iteration: int,
        bound: float,
        conv2d: nn.Conv2d,
    ):
        super().__init__()
        self.iteration = iteration
        self.bound = bound
        self.conv2d = conv2d

        self.u = None
        self.v = None
        self.padding = None
        self.agg_padding = None
        self.output_padding = None

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
        return getattr(self.conv2d, name)

    def initialize_uv(self, inputs):
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        in_channels = self.conv2d.in_channels
        self.in_shape = (1, in_channels, in_height, in_width)
        out_height = in_height // self.conv2d.stride[0]
        out_width = in_width // self.conv2d.stride[1]
        out_channels = self.conv2d.out_channels
        self.out_shape = (1, out_channels, out_height, out_width)
        device = self.conv2d.weight.device

        self.v = nn.Parameter(
            torch.randn(self.in_shape, device=device), requires_grad=False
        )
        self.u = nn.Parameter(
            torch.randn(self.out_shape, device=device), requires_grad=False
        )

    def forward(self, inputs):
        with torch.no_grad():
            if self.u is None:
                self.initialize_uv(inputs)
                self.padding = self.calc_same_padding(
                    self.u.shape,
                    self.conv2d.weight.shape,
                    self.conv2d.stride,
                    self.conv2d.dilation,
                )
                self.agg_padding = (
                    ceil((self.padding[0] + self.padding[1]) / 2),
                    ceil((self.padding[2] + self.padding[3]) / 2),
                    # (self.padding[0] + self.padding[1]) // 2,
                    # (self.padding[2] + self.padding[3]) // 2
                )
                self.output_padding = self.calc_output_padding(
                    inputs=self.u,
                    output_size=self.v.shape,
                    stride=self.conv2d.stride,
                    padding=self.agg_padding,
                    kernel_size=self.conv2d.kernel_size,
                    dilation=self.conv2d.dilation,
                )

            self.update_weight()

        output = self.conv2d(inputs)

        with torch.no_grad():
            self.restore_weight()

        return output

    @staticmethod
    def calc_output_padding(
        inputs, output_size, stride, padding, kernel_size, dilation
    ):
        num_spatial_dims = 2

        has_batch_dim = inputs.dim() == num_spatial_dims + 2
        num_non_spatial_dims = 2 if has_batch_dim else 1
        if len(output_size) == num_non_spatial_dims + num_spatial_dims:
            output_size = output_size[num_non_spatial_dims:]
        if len(output_size) != num_spatial_dims:
            raise ValueError(
                (
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} "
                    "elements (got {})"
                ).format(
                    num_spatial_dims,
                    inputs.dim(),
                    num_spatial_dims,
                    num_non_spatial_dims + num_spatial_dims,
                    len(output_size),
                )
            )

        min_sizes = []
        max_sizes = []
        for d in range(num_spatial_dims):
            dim_size = (
                (inputs.size(d + num_non_spatial_dims) - 1) * stride[d]
                - 2 * padding[d]
                + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1)
                + 1
            )
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(
                    f"requested an output size of {output_size}, but valid sizes range "
                    f"from {min_sizes} to {max_sizes} "
                    f"(for an input of {inputs.size()[2:]})"
                )

        res = []
        for d in range(num_spatial_dims):
            res.append(output_size[d] - min_sizes[d])

        return tuple(res)

    @staticmethod
    def calc_same_padding(input_shape, filter_shape, stride, dilation):
        """Calculates padding values for 'SAME' padding for conv2d.

        Args:
            input_shape (tuple or list): Shape of the input data.
                [batch, channels, height, width]
            filter_shape (tuple or list): Shape of the filter/kernel.
                [out_channels, in_channels, kernel_height, kernel_width]
            stride (int or tuple): Stride of the convolution operation.
            dilation (int or tuple): Dilation rate of the convolution operation.

        Returns:
            padding (tuple): Tuple representing padding
                (padding_left, padding_right, padding_top, padding_bottom)
        """
        if isinstance(stride, int):
            stride_height = stride_width = stride
        else:
            stride_height, stride_width = stride

        if isinstance(dilation, int):
            dilation_height, dilation_width = dilation, dilation
        else:
            dilation_height, dilation_width = dilation

        in_height, in_width = input_shape[2], input_shape[3]
        filter_height, filter_width = filter_shape[2], filter_shape[3]

        effective_filter_height = filter_height + (filter_height - 1) * (
            dilation_height - 1
        )
        effective_filter_width = filter_width + (filter_width - 1) * (
            dilation_width - 1
        )

        if in_height % stride_height == 0:
            pad_along_height = max(effective_filter_height - stride_height, 0)
        else:
            pad_along_height = max(
                effective_filter_height - (in_height % stride_height), 0
            )

        if in_width % stride_width == 0:
            pad_along_width = max(effective_filter_width - stride_width, 0)
        else:
            pad_along_width = max(effective_filter_width - (in_width % stride_width), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    def update_weight(self):
        u_hat = self.u
        v_hat = self.v

        if self.training:
            for _ in range(self.iteration):
                v_hat = F.conv_transpose2d(
                    input=u_hat,
                    weight=self.conv2d.weight,
                    bias=self.conv2d.bias,
                    stride=self.conv2d.stride,
                    padding=self.agg_padding,
                    output_padding=self.output_padding,
                    groups=self.conv2d.groups,
                    dilation=self.conv2d.dilation,
                )
                v_hat = F.normalize(v_hat.reshape(1, -1)).reshape(v_hat.shape)

                v_hat_pad = F.pad(v_hat, self.padding)
                u_hat = F.conv2d(
                    input=v_hat_pad,
                    weight=self.conv2d.weight,
                    bias=self.conv2d.bias,
                    stride=self.conv2d.stride,
                    dilation=self.conv2d.dilation,
                    groups=self.conv2d.groups,
                )
                u_hat = F.normalize(u_hat.reshape(1, -1)).reshape(u_hat.shape)
        else:
            v_hat_pad = F.pad(v_hat, self.padding)

        v_w_hat = F.conv2d(
            input=v_hat_pad,
            weight=self.conv2d.weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )

        sigma = v_w_hat.flatten() @ u_hat.flatten()

        if self.bound < sigma:
            normed_weight = self.bound / sigma * self.conv2d.weight
        else:
            normed_weight = self.conv2d.weight

        self.u.copy_(u_hat)
        self.v.copy_(v_hat)
        self.saved_weight = self.conv2d.weight.clone().detach()
        self.conv2d.weight.data = normed_weight  # Only way to circumvent autograd

    def restore_weight(self):
        self.conv2d.weight.data = self.saved_weight


class SpectralNormalizedConv2d(nn.Module):
    def __init__(
        self,
        iteration: int,
        bound: float,
        conv2d: nn.Conv2d,
    ):
        super().__init__()
        self.iteration = iteration
        self.bound = bound
        self.conv2d = conv2d

        self.u = None
        self.v = None
        self.padding = None
        self.agg_padding = None
        self.output_padding = None

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
        return getattr(self.conv2d, name)

    def initialize_uv(self, inputs):
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        in_channels = self.conv2d.in_channels
        self.in_shape = (1, in_channels, in_height, in_width)
        out_height = in_height // self.conv2d.stride[0]
        out_width = in_width // self.conv2d.stride[1]
        out_channels = self.conv2d.out_channels
        self.out_shape = (1, out_channels, out_height, out_width)
        device = self.conv2d.weight.device

        self.v = nn.Parameter(
            torch.randn(self.in_shape, device=device), requires_grad=False
        )
        self.u = nn.Parameter(
            torch.randn(self.out_shape, device=device), requires_grad=False
        )

    def forward(self, inputs):
        with torch.no_grad():
            if self.u is None:
                self.initialize_uv(inputs)
                self.padding = self.calc_same_padding(
                    self.u.shape,
                    self.conv2d.weight.shape,
                    self.conv2d.stride,
                    self.conv2d.dilation,
                )
                self.agg_padding = (
                    ceil((self.padding[0] + self.padding[1]) / 2),
                    ceil((self.padding[2] + self.padding[3]) / 2),
                )
                self.output_padding = self.calc_output_padding(
                    inputs=self.u,
                    output_size=self.v.shape,
                    stride=self.conv2d.stride,
                    padding=self.agg_padding,
                    kernel_size=self.conv2d.kernel_size,
                    dilation=self.conv2d.dilation,
                )

            factor = self.normalization_factor()

        normed_weight = factor * self.conv2d.weight

        output = F.conv2d(
            input=inputs,
            weight=normed_weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )

        return output

    @staticmethod
    def calc_output_padding(
        inputs, output_size, stride, padding, kernel_size, dilation
    ):
        num_spatial_dims = 2

        has_batch_dim = inputs.dim() == num_spatial_dims + 2
        num_non_spatial_dims = 2 if has_batch_dim else 1
        if len(output_size) == num_non_spatial_dims + num_spatial_dims:
            output_size = output_size[num_non_spatial_dims:]
        if len(output_size) != num_spatial_dims:
            raise ValueError(
                (
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} "
                    "elements (got {})"
                ).format(
                    num_spatial_dims,
                    inputs.dim(),
                    num_spatial_dims,
                    num_non_spatial_dims + num_spatial_dims,
                    len(output_size),
                )
            )

        min_sizes = []
        max_sizes = []
        for d in range(num_spatial_dims):
            dim_size = (
                (inputs.size(d + num_non_spatial_dims) - 1) * stride[d]
                - 2 * padding[d]
                + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1)
                + 1
            )
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(
                    f"requested an output size of {output_size}, but valid sizes range "
                    f"from {min_sizes} to {max_sizes} "
                    f"(for an input of {inputs.size()[2:]})"
                )

        res = []
        for d in range(num_spatial_dims):
            res.append(output_size[d] - min_sizes[d])

        return tuple(res)

    @staticmethod
    def calc_same_padding(input_shape, filter_shape, stride, dilation):
        """Calculates padding values for 'SAME' padding for conv2d.

        Args:
            input_shape (tuple or list): Shape of the input data.
                [batch, channels, height, width]
            filter_shape (tuple or list): Shape of the filter/kernel.
                [out_channels, in_channels, kernel_height, kernel_width]
            stride (int or tuple): Stride of the convolution operation.
            dilation (int or tuple): Dilation rate of the convolution operation.

        Returns:
            padding (tuple): Tuple representing padding
                (padding_left, padding_right, padding_top, padding_bottom)
        """
        if isinstance(stride, int):
            stride_height = stride_width = stride
        else:
            stride_height, stride_width = stride

        if isinstance(dilation, int):
            dilation_height, dilation_width = dilation, dilation
        else:
            dilation_height, dilation_width = dilation

        in_height, in_width = input_shape[2], input_shape[3]
        filter_height, filter_width = filter_shape[2], filter_shape[3]

        effective_filter_height = filter_height + (filter_height - 1) * (
            dilation_height - 1
        )
        effective_filter_width = filter_width + (filter_width - 1) * (
            dilation_width - 1
        )

        if in_height % stride_height == 0:
            pad_along_height = max(effective_filter_height - stride_height, 0)
        else:
            pad_along_height = max(
                effective_filter_height - (in_height % stride_height), 0
            )

        if in_width % stride_width == 0:
            pad_along_width = max(effective_filter_width - stride_width, 0)
        else:
            pad_along_width = max(effective_filter_width - (in_width % stride_width), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    def normalization_factor(self):
        u_hat = self.u
        v_hat = self.v

        if self.training:
            for _ in range(self.iteration):
                v_hat = F.conv_transpose2d(
                    input=u_hat,
                    weight=self.conv2d.weight,
                    bias=self.conv2d.bias,
                    stride=self.conv2d.stride,
                    padding=self.agg_padding,
                    output_padding=self.output_padding,
                    groups=self.conv2d.groups,
                    dilation=self.conv2d.dilation,
                )
                v_hat = F.normalize(v_hat.reshape(1, -1)).reshape(v_hat.shape)

                v_hat_pad = F.pad(v_hat, self.padding)
                u_hat = F.conv2d(
                    input=v_hat_pad,
                    weight=self.conv2d.weight,
                    bias=self.conv2d.bias,
                    stride=self.conv2d.stride,
                    dilation=self.conv2d.dilation,
                    groups=self.conv2d.groups,
                )
                u_hat = F.normalize(u_hat.reshape(1, -1)).reshape(u_hat.shape)
        else:
            v_hat_pad = F.pad(v_hat, self.padding)

        v_w_hat = F.conv2d(
            input=v_hat_pad,
            weight=self.conv2d.weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )

        sigma = v_w_hat.flatten() @ u_hat.flatten()

        if self.bound < sigma:
            factor = self.bound / sigma
        else:
            factor = 1

        self.u.copy_(u_hat)
        self.v.copy_(v_hat)

        return factor


class SpectralNormalizedConv2dDUE(nn.Module):
    def __init__(
        self,
        iteration: int,
        bound: float,
        conv2d: nn.Conv2d,
    ):
        super().__init__()
        self.iteration = iteration
        self.bound = bound
        self.conv2d = conv2d

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
        return getattr(self.conv2d, name)

    def initialize_uv(self, inputs):
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        in_channels = self.conv2d.in_channels
        self.in_shape = (1, in_channels, in_height, in_width)
        out_height = in_height // self.conv2d.stride[0]
        out_width = in_width // self.conv2d.stride[1]
        out_channels = self.conv2d.out_channels
        self.out_shape = (1, out_channels, out_height, out_width)
        device = self.conv2d.weight.device

        self.v = nn.Parameter(
            F.normalize(
                torch.randn(self.in_shape, device=device).flatten(), dim=0
            ).reshape(self.in_shape),
            requires_grad=False,
        )
        self.u = nn.Parameter(
            F.normalize(
                torch.randn(self.out_shape, device=device).flatten(), dim=0
            ).reshape(self.out_shape),
            requires_grad=False,
        )

    def forward(self, inputs):
        with torch.no_grad():
            if not hasattr(self, "u"):
                self.initialize_uv(inputs)

        factor = self.normalization_factor()
        normed_weight = factor * self.conv2d.weight

        output = F.conv2d(
            input=inputs,
            weight=normed_weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )

        return output

    def normalization_factor(self):
        u = self.u
        v = self.v

        if self.training:
            with torch.no_grad():
                output_padding = 0
                if self.conv2d.stride[0] > 1:
                    output_padding = 1 - self.in_shape[-1] % 2

                for _ in range(self.iteration):
                    v_s = F.conv_transpose2d(
                        input=u,
                        weight=self.conv2d.weight,
                        bias=self.conv2d.bias,
                        stride=self.conv2d.stride,
                        padding=self.conv2d.padding,
                        output_padding=output_padding,
                        groups=self.conv2d.groups,
                        dilation=self.conv2d.dilation,
                    )
                    v = F.normalize(v_s.flatten(), dim=0).reshape(v_s.shape)

                    u_s = F.conv2d(
                        input=v,
                        weight=self.conv2d.weight,
                        bias=self.conv2d.bias,
                        stride=self.conv2d.stride,
                        padding=self.conv2d.padding,
                        dilation=self.conv2d.dilation,
                        groups=self.conv2d.groups,
                    )
                    u = F.normalize(u_s.flatten(), dim=0).reshape(u_s.shape)

        weight_v = F.conv2d(
            input=v,
            weight=self.conv2d.weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )

        sigma = weight_v.flatten() @ u.flatten()

        if self.bound < sigma:
            factor = self.bound / sigma
        else:
            factor = 1

        self.u.copy_(u)
        self.v.copy_(v)

        return factor


class SNGPWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates an SNGP from it.
    """

    def __init__(
        self,
        model: nn.Module,
        is_spectral_normalized: bool,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        sngp_version: str,
        num_mc_samples: int,
        num_random_features: int,
        gp_kernel_scale: float,
        gp_output_bias: float,
        gp_random_feature_type: str,
        is_gp_input_normalized: bool,
        gp_cov_momentum: float,
        gp_cov_ridge_penalty: float,
        gp_input_dim,
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
            if sngp_version == "google":
                conv = SpectralNormalizedConv2dGoogle
            elif sngp_version == "original":
                conv = SpectralNormalizedConv2d
            elif sngp_version == "due":
                conv = SpectralNormalizedConv2dDUE
            else:
                raise ValueError("Invalid version provided")

            SNC = partial(
                conv,
                spectral_normalization_iteration,
                spectral_normalization_bound,
            )
            replace(model, "Conv2d", SNC)

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
