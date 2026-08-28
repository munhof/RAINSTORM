from __future__ import annotations

from pkgutil import extend_path


__version__ = "1.0.12"
__author__ = "Santiago D'hers"
__email__ = "sdhers@fbmc.fcen.uba.ar"

__path__ = extend_path(__path__, __name__)

_backend_path = __path__[0] + "/backend"
if _backend_path not in __path__:
    __path__.append(_backend_path)


__all__ = ["__version__", "__author__", "__email__"]
