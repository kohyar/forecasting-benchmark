"""Model registry.

Adding a model is one adapter plus one register() call. The runner resolves
names through here and never learns any of them.
"""
import importlib

from tsbench.models.base import FAMILIES, ModelAdapter

#: Every adapter module. torch is imported first: it and lightgbm each bundle
#: an OpenMP runtime, and loading lightgbm's first breaks torch.
ADAPTER_MODULES = (
    "tsbench.models.statsforecast_adapters",
    "tsbench.models.lightgbm_adapters",
    "tsbench.models.prophet_adapter",
    "tsbench.models.neural_adapters",
    "tsbench.models.gated",
)


class Registry:
    def __init__(self):
        self._entries = {}

    def register(self, adapter: type, enabled: bool = True, disabled_reason: str = ""):
        name = adapter.name
        if not name:
            raise ValueError(f"{adapter.__name__} must declare a name")
        if adapter.family not in FAMILIES:
            raise ValueError(
                f"{name}: family {adapter.family!r} must be one of {FAMILIES}")
        if name in self._entries:
            raise ValueError(f"{name} is already registered")
        if not enabled and not disabled_reason:
            raise ValueError(f"{name}: a disabled adapter must give a reason")

        self._entries[name] = {
            "adapter": adapter,
            "enabled": enabled,
            "disabled_reason": disabled_reason,
        }
        return adapter

    def get(self, name: str) -> type:
        if name not in self._entries:
            raise KeyError(f"unknown model {name!r}; registered: {sorted(self._entries)}")
        return self._entries[name]["adapter"]

    def create(self, name: str, cfg, params: dict | None = None, device: str = "cpu") -> ModelAdapter:
        return self.get(name)(cfg, params=params, device=device)

    def names(self) -> list:
        return sorted(self._entries)

    def available_names(self) -> list:
        return [n for n in self.names() if self.status(n)["available"]]

    def status(self, name: str) -> dict:
        entry = self._entries[name] if name in self._entries else None
        if entry is None:
            raise KeyError(f"unknown model {name!r}; registered: {sorted(self._entries)}")

        adapter = entry["adapter"]
        enabled, reason = entry["enabled"], entry["disabled_reason"]
        if enabled:
            ok, why = adapter.is_available()
            if not ok:
                enabled, reason = False, why
        return {
            "name": name,
            "family": adapter.family,
            "enabled": entry["enabled"],
            "available": enabled,
            "disabled_reason": reason,
        }

    def report(self) -> list:
        return [self.status(n) for n in self.names()]


_DEFAULT = Registry()
_POPULATED = False


def default() -> Registry:
    """The registry with every adapter loaded.

    Only the parent process should call this; a worker imports the single
    module it needs so that it does not pull in a conflicting library.
    """
    global _POPULATED
    if not _POPULATED:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        for module in ADAPTER_MODULES:
            importlib.import_module(module)
        _POPULATED = True
    return _DEFAULT


def register(adapter: type = None, *, enabled: bool = True, disabled_reason: str = ""):
    """Decorator form used by the adapter modules."""
    def wrap(cls):
        return _DEFAULT.register(cls, enabled=enabled, disabled_reason=disabled_reason)
    return wrap(adapter) if adapter is not None else wrap
