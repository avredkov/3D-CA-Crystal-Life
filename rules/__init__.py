"""
Rules plugin package.

Each ruleset should be a module in this package exposing:

    def assign_rules(Ru: np.ndarray, params: dict[str, float]) -> None

This function must fill the provided Ru array in-place using a mapping of
parameter keys to values in [0, 1].
"""

__all__ = [
    # modules can be added here for discoverability if desired
]


