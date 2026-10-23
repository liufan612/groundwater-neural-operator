from .operators import build_model_factory
from .unet import UNetSmall

NAMES = ("unet", "fno", "uno", "afno", "deeponet")

def build_operator(name, input_channels, output_channels=2, model_config=None):
    kwargs = dict(model_config or {})
    if name == "unet":
        return UNetSmall(input_channels, output_channels, **kwargs)
    return build_model_factory(name, kwargs)(input_channels, output_channels, 30, 50)
