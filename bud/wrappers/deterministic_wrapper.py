from torch import nn

from bud.wrappers.model_wrapper import PosteriorWrapper


class DeterministicWrapper(PosteriorWrapper):
    """
    This module takes a model as input and keeps it deterministic. It only serves as
    connective tissue to the rest of the framework.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)
