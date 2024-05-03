import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from bud.layers import set_layer_config
from bud.wrappers import (
    CorrectnessPredictionWrapper,
    DeepCorrectnessPredictionWrapper,
    DeepEnsembleWrapper,
    DeepLossPredictionWrapper,
    DeterministicWrapper,
    TemperatureWrapper,
    DropoutWrapper,
    DUQWrapper,
    DDUWrapper,
    HETXLWrapper,
    LaplaceWrapper,
    MahalanobisWrapper,
    MCInfoNCEWrapper,
    NonIsotropicvMFWrapper,
    LossPredictionWrapper,
    ShallowEnsembleWrapper,
    SNGPWrapper,
    EDLWrapper,
)

from ._helpers import load_checkpoint
from ._hub import load_model_config_from_hf
from ._pretrained import PretrainedCfg
from ._registry import is_model, model_entrypoint, split_model_name_tag

__all__ = ["parse_model_name", "safe_model_name", "create_model"]

VALID_WRAPPERS = [
    "correctness-prediction",
    "deep-ensemble",
    "deterministic",
    "dropout",
    "duq",
    "ddu",
    "het-xl",
    "laplace",
    "mahalanobis",
    "mcinfonce",
    "non-isotropic-vmf",
    "risk-prediction",
    "shallow-ensemble",
    "sngp",
]


def parse_model_name(model_name: str):
    if model_name.startswith("hf_hub"):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace("hf_hub", "hf-hub")
    parsed = urlsplit(model_name)
    assert parsed.scheme in ("", "timm", "hf-hub")
    if parsed.scheme == "hf-hub":
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return "timm", model_name


def safe_model_name(model_name: str, remove_source: bool = True):
    # return a filename / path safe model name
    def make_safe(name):
        return "".join(c if c.isalnum() else "_" for c in name).rstrip("_")

    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
    model_name: str,
    model_wrapper_name: str,
    pretrained: bool,
    scriptable: Optional[bool],
    weight_paths,
    num_hidden_features,
    is_reset_classifier,
    mlp_depth,
    stopgrad,
    num_hooks,
    module_type,
    module_name_regex,
    dropout_probability,
    is_filterwise_dropout,
    num_mc_samples,
    rbf_length_scale,
    ema_momentum,
    matrix_rank,
    temperature,
    is_last_layer_laplace,
    pred_type,
    prior_optimization_method,
    hessian_structure,
    link_approx,
    magnitude,
    initial_average_kappa,
    num_heads,
    is_spectral_normalized,
    spectral_normalization_iteration,
    spectral_normalization_bound,
    is_batch_norm_spectral_normalized,
    use_tight_norm_for_pointwise_convs,
    num_random_features,
    gp_kernel_scale,
    gp_output_bias,
    gp_random_feature_type,
    is_gp_input_normalized,
    gp_cov_momentum,
    gp_cov_ridge_penalty,
    gp_input_dim,
    use_pretrained,
    checkpoint_path: str,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    **kwargs,
):
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    <Tip>
        **kwargs will be passed through entrypoint fn to
        ``timm.models.build_model_with_cfg()`` and then the model class __init__().
        kwargs values set to None are pruned before passing.
    </Tip>

    Args:
        model_name: Name of model to instantiate.
        model_wrapper_name: Model wrapper used to obtain uncertainty estimates.
        num_heads: Number of heads for a shallow ensemble.
        loss_regressor_width: With of loss-regressor module.
        loss_regressor_depth: Number of hidden layers of loss-regressor module.
        loss_regressor_feature_depth: Depth of features [0, 1] in the network to attach
            the loss-regressor to.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        scriptable: Set layer config so that model is jit scriptable (not working for all
            models yet).
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.
        exportable: Set layer config so that model is traceable / ONNX exportable (not
            fully impl/obeyed yet).
        no_jit: Set layer config so that model doesn't utilize jit scripted layers (so
            far activations only).

    Keyword Args:
        drop_rate (float): Classifier dropout rate for training.
        drop_path_rate (float): Stochastic depth drop rate for training.
        global_pool (str): Classifier global pooling type.

    Example:

    ```py
    >>> from timm import create_model

    >>> # Create a MobileNetV3-Large model with no pretrained weights.
    >>> model = create_model('mobilenetv3_large_100')

    >>> # Create a MobileNetV3-Large model with pretrained weights.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True)
    >>> model.num_classes
    1000

    >>> # Create a MobileNetV3-Large model with pretrained weights
    >>> # and a new head with 10 classes.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
    >>> model.num_classes
    10
    ```
    """
    # Parameters that aren't supported by all models or are intended to only override
    # model defaults if set should default to None in command line args/cfg.
    # Remove them if they are present and not set so that non-supporting models don't
    # break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        assert (
            not pretrained_cfg
        ), "pretrained_cfg should not be set when sourcing model from Hugging Face Hub."
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if pretrained_tag and not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError(f"Unknown model ({model_name})")

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if is_reset_classifier:
        model.reset_classifier(model.num_classes)

    model = wrap_model(
        model=model,
        model_wrapper_name=model_wrapper_name,
        weight_paths=weight_paths,
        num_hidden_features=num_hidden_features,
        mlp_depth=mlp_depth,
        stopgrad=stopgrad,
        num_hooks=num_hooks,
        module_type=module_type,
        module_name_regex=module_name_regex,
        dropout_probability=dropout_probability,
        is_filterwise_dropout=is_filterwise_dropout,
        num_mc_samples=num_mc_samples,
        rbf_length_scale=rbf_length_scale,
        ema_momentum=ema_momentum,
        matrix_rank=matrix_rank,
        temperature=temperature,
        is_last_layer_laplace=is_last_layer_laplace,
        pred_type=pred_type,
        prior_optimization_method=prior_optimization_method,
        hessian_structure=hessian_structure,
        link_approx=link_approx,
        magnitude=magnitude,
        initial_average_kappa=initial_average_kappa,
        num_heads=num_heads,
        is_spectral_normalized=is_spectral_normalized,
        spectral_normalization_iteration=spectral_normalization_iteration,
        spectral_normalization_bound=spectral_normalization_bound,
        is_batch_norm_spectral_normalized=is_batch_norm_spectral_normalized,
        use_tight_norm_for_pointwise_convs=use_tight_norm_for_pointwise_convs,
        num_random_features=num_random_features,
        gp_kernel_scale=gp_kernel_scale,
        gp_output_bias=gp_output_bias,
        gp_random_feature_type=gp_random_feature_type,
        is_gp_input_normalized=is_gp_input_normalized,
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        gp_input_dim=gp_input_dim,
        use_pretrained=use_pretrained,
        kwargs=kwargs,
    )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model


def wrap_model(
    model,
    model_wrapper_name,
    weight_paths,
    num_hidden_features,
    mlp_depth,
    stopgrad,
    num_hooks,
    module_type,
    module_name_regex,
    dropout_probability,
    is_filterwise_dropout,
    num_mc_samples,
    rbf_length_scale,
    ema_momentum,
    matrix_rank,
    temperature,
    is_last_layer_laplace,
    pred_type,
    prior_optimization_method,
    hessian_structure,
    link_approx,
    magnitude,
    initial_average_kappa,
    num_heads,
    is_spectral_normalized,
    spectral_normalization_iteration,
    spectral_normalization_bound,
    is_batch_norm_spectral_normalized,
    use_tight_norm_for_pointwise_convs,
    num_random_features,
    gp_kernel_scale,
    gp_output_bias,
    gp_random_feature_type,
    is_gp_input_normalized,
    gp_cov_momentum,
    gp_cov_ridge_penalty,
    gp_input_dim,
    use_pretrained,
    kwargs,
):
    if model_wrapper_name == "correctness-prediction":
        wrapped_model = CorrectnessPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
        )
    elif model_wrapper_name == "deep-correctness-prediction":
        wrapped_model = DeepCorrectnessPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "temperature":
        wrapped_model = TemperatureWrapper(model=model, weight_path=weight_paths[0])
    elif model_wrapper_name == "deep-ensemble":
        return DeepEnsembleWrapper(
            model=model,
            weight_paths=weight_paths,
            use_pretrained=use_pretrained,
            kwargs=kwargs,
        )
    elif model_wrapper_name == "deterministic":
        wrapped_model = DeterministicWrapper(model=model)
    elif model_wrapper_name == "dropout":
        wrapped_model = DropoutWrapper(
            model=model,
            dropout_probability=dropout_probability,
            is_filterwise_dropout=is_filterwise_dropout,
            num_mc_samples=num_mc_samples,
        )
    elif model_wrapper_name == "duq":
        wrapped_model = DUQWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            rbf_length_scale=rbf_length_scale,
            ema_momentum=ema_momentum,
        )
    elif model_wrapper_name == "ddu":
        wrapped_model = DDUWrapper(
            model=model,
            is_spectral_normalized=is_spectral_normalized,
            spectral_normalization_iteration=spectral_normalization_iteration,
            spectral_normalization_bound=spectral_normalization_bound,
            is_batch_norm_spectral_normalized=is_batch_norm_spectral_normalized,
            use_tight_norm_for_pointwise_convs=use_tight_norm_for_pointwise_convs,
        )
    elif model_wrapper_name == "het-xl":
        wrapped_model = HETXLWrapper(
            model=model,
            matrix_rank=matrix_rank,
            num_mc_samples=num_mc_samples,
            temperature=temperature,
        )
    elif model_wrapper_name == "laplace":
        wrapped_model = LaplaceWrapper(
            model=model,
            num_mc_samples=num_mc_samples,
            weight_path=weight_paths[0],
            is_last_layer_laplace=is_last_layer_laplace,
            pred_type=pred_type,
            prior_optimization_method=prior_optimization_method,
            hessian_structure=hessian_structure,
            link_approx=link_approx,
        )
    elif model_wrapper_name == "mahalanobis":
        wrapped_model = MahalanobisWrapper(
            model=model,
            magnitude=magnitude,
            weight_path=weight_paths[0],
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "mcinfonce":
        wrapped_model = MCInfoNCEWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            initial_average_kappa=initial_average_kappa,
        )
    elif model_wrapper_name == "non-isotropic-vmf":
        wrapped_model = NonIsotropicvMFWrapper(
            model=model,
            num_mc_samples=num_mc_samples,
            num_hidden_features=num_hidden_features,
            initial_average_kappa=initial_average_kappa,
        )
    elif model_wrapper_name == "risk-prediction":
        wrapped_model = LossPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
        )
    elif model_wrapper_name == "edl":
        wrapped_model = EDLWrapper(model=model)
    elif model_wrapper_name == "deep-risk-prediction":
        wrapped_model = DeepLossPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "shallow-ensemble":
        wrapped_model = ShallowEnsembleWrapper(model=model, num_heads=num_heads)
    elif model_wrapper_name == "sngp":
        wrapped_model = SNGPWrapper(
            model=model,
            is_spectral_normalized=is_spectral_normalized,
            use_tight_norm_for_pointwise_convs=use_tight_norm_for_pointwise_convs,
            spectral_normalization_iteration=spectral_normalization_iteration,
            spectral_normalization_bound=spectral_normalization_bound,
            is_batch_norm_spectral_normalized=is_batch_norm_spectral_normalized,
            num_mc_samples=num_mc_samples,
            num_random_features=num_random_features,
            gp_kernel_scale=gp_kernel_scale,
            gp_output_bias=gp_output_bias,
            gp_random_feature_type=gp_random_feature_type,
            is_gp_input_normalized=is_gp_input_normalized,
            gp_cov_momentum=gp_cov_momentum,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            gp_input_dim=gp_input_dim,
        )
    else:
        raise ValueError(
            f'Model wrapper "{model_wrapper_name}" is currently not implemented or you '
            f"made a typo. Valid options are {VALID_WRAPPERS}."
        )

    return wrapped_model
