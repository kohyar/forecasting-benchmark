"""Adapters registered but held back, so enabling one is a config change
rather than a code change."""
from tsbench.models.base import ModelAdapter
from tsbench.models.registry import register


@register(enabled=False,
          disabled_reason="licence not yet cleared for benchmark publication")
class TiRexAdapter(ModelAdapter):
    """NX-AI TiRex, zero-shot. Excluded pending the licence check; flip
    `enabled=True` once resolved and nothing else needs to change."""

    name = "tirex"
    family = "foundation"

    @classmethod
    def is_available(cls):
        try:
            import tirex  # noqa: F401
        except ImportError as exc:
            return False, f"tirex is not installed ({exc})"
        return True, ""

    def fit(self, train_df):
        self._ids = sorted(train_df["unique_id"].unique())
        self._last = train_df["ds"].max()

    def predict(self, horizon):
        raise NotImplementedError("TiRex adapter is gated on its licence")

    def supports_quantiles(self) -> bool:
        return True
