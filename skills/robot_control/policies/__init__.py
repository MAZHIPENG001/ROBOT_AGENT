"""可扩展策略集合，OpenPI 只是一个策略实现。"""
__all__ = ["OpenPiPolicy"]


def __getattr__(name: str):
    if name == "OpenPiPolicy":
        from .openpi import OpenPiPolicy
        return OpenPiPolicy
    raise AttributeError(name)
