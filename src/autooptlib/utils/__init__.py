"""Utility namespace for AutoOptLib."""

try:  # noqa: SIM105
    from . import select  # noqa: F401
except Exception:  # pylint: disable=broad-except
    select = None  # type: ignore

__all__ = [
    name for name in ("select",) if name in globals() and globals()[name] is not None
]
