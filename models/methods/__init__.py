from .dyn_hsi import DynHSIModel
from .original_hsi import FaithfulLingoModel, FaithfulTrumansModel


def build_hsi_method_model(method: str, motion_dim: int = 84, **kwargs):
    key = str(method).lower()
    if key == "lingo":
        return FaithfulLingoModel(motion_dim=motion_dim, **kwargs)
    if key == "trumans":
        return FaithfulTrumansModel(motion_dim=motion_dim, **kwargs)
    if key == "dyn_hsi":
        return DynHSIModel(motion_dim=motion_dim, **kwargs)
    raise ValueError(f"unknown HSI method: {method}")


__all__ = ["DynHSIModel", "FaithfulLingoModel", "FaithfulTrumansModel", "build_hsi_method_model"]
