"""仿真/真机适配器。"""
__all__ = ["LiberoAdapter"]


def __getattr__(name: str):
    if name == "LiberoAdapter":
        from .libero import LiberoAdapter
        return LiberoAdapter
    raise AttributeError(name)
