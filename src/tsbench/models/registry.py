"""Model registry.

Adding a model is one adapter plus one register() call. The runner resolves
names through here and never learns any of them.
"""
from tsbench.models.base import FAMILIES, ModelAdapter


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


def default() -> Registry:
    """The populated registry. Importing tsbench.models fills it."""
    import tsbench.models  # noqa: F401  (registration side effect)
    return _DEFAULT


def register(adapter: type = None, *, enabled: bool = True, disabled_reason: str = ""):
    """Decorator form used by the adapter modules."""
    def wrap(cls):
        return _DEFAULT.register(cls, enabled=enabled, disabled_reason=disabled_reason)
    return wrap(adapter) if adapter is not None else wrap
