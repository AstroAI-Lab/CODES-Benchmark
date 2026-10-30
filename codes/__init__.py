from importlib import metadata

from .benchmark import *
from .finetune import *
from .surrogates import *
from .train import *
from .tune import *
from .utils import *

try:
    __version__ = metadata.version("codes")
except metadata.PackageNotFoundError:  # Imported from a source tree without installing.
    __version__ = "unknown"
