from joblib import register_parallel_backend
from joblib_modal._modal import ModalBackend

register_parallel_backend("modal", ModalBackend)

__version__ = "0.1.0"

__all__ = ["ModalBackend", "__version__"]
