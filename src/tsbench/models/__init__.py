"""Model adapters.

Deliberately empty: importing `tsbench.models.<module>` must register that
adapter and nothing else. Worker processes rely on it - lightgbm and torch
each bundle an OpenMP runtime and segfault when both are exercised in one
process, so a lightgbm worker must never load torch. Use
`registry.default()` when you want them all.
"""
