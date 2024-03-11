from bud.wrappers.model_wrapper import PosteriorWrapper
from urllib.parse import urlsplit
import os
from bud.models._hub import load_model_config_from_hf
from bud.models._registry import is_model, model_entrypoint, split_model_name_tag
from bud.layers import set_layer_config


class DeepEnsembleWrapper(PosteriorWrapper):
    """
    Wrapper to manage an ensemble of independently trained models.
    """

    def __init__(
        self,
        model,
        weight_paths: list,
        use_pretrained: bool,
        kwargs: dict,
    ):
        super().__init__(model=model)
        self.weight_paths = weight_paths
        self.weight_path = self.weight_paths[0]
        self.num_models = len(weight_paths)

        self.use_pretrained = use_pretrained
        self.kwargs = kwargs

        self.load_model(0)

    def load_model(self, index: int):
        """
        Load a model based on the index.
        """
        if index < 0 or index >= self.num_models:
            raise ValueError("Index out of bounds")

        if not self.use_pretrained:
            self.weight_path = self.weight_paths[index]
            super().load_model()
        else:
            del self.model
            model_name = self.weight_paths[index]
            self.model = self.create_entrypoint(model_name)

    def create_entrypoint(self, model_name: str):
        kwargs = {k: v for k, v in self.kwargs.items() if v is not None}
        pretrained_cfg = None

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
        with set_layer_config(scriptable=False, exportable=False, no_jit=None):
            model = create_fn(
                pretrained=True,
                pretrained_cfg=pretrained_cfg,
                **kwargs,
            )

        return model


def parse_model_name(model_name: str):
    if model_name.startswith("hf_hub"):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace("hf_hub", "hf-hub")
    parsed = urlsplit(model_name)
    assert parsed.scheme in ("", "timm", "hf-hub")
    if parsed.scheme == "hf-hub":
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return "timm", model_name
