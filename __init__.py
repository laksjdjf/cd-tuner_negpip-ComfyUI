from .cd_tuner import CDTuner
from .negpip import Negpip

NODE_CLASS_MAPPINGS = {
    "CDTuner": CDTuner,
    "Negapip": Negpip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CDTuner": "Apply CDTuner",
    "Negapip": "Apply Negapip",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]