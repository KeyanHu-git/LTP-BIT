_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from .metrics import pixel_wise, perceptual, distributional  # noqa: F401
    import Dataset.datasets  # noqa: F401
    _REGISTERED = True
