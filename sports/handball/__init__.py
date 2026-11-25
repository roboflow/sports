from sports.handball.config import CourtConfiguration, HandballCourtConfiguration

__all__ = [
    "CourtConfiguration",
    "HandballCourtConfiguration",
    "draw_court",
    "draw_paths_on_court",
    "draw_points_on_court",
]


def __getattr__(name):
    if name in {"draw_court", "draw_paths_on_court", "draw_points_on_court"}:
        from sports.handball import annotators

        return getattr(annotators, name)
    raise AttributeError(f"module 'sports.handball' has no attribute {name!r}")
