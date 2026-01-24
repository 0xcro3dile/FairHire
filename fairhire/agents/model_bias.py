# Fairlearn wrapper for model prediction bias
# L20 compliant: strict type hints
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from numpy.typing import NDArray


class ModelBiasAgent:
    def __init__(self, sensitive_features: list[str] | None = None) -> None:
        self.sensitive_features = sensitive_features or []

    def compute_metrics(
        self,
        y_true: NDArray[np.int_],
        y_pred: NDArray[np.int_],
        sensitive: NDArray[np.int_],
    ) -> dict[str, float]:
        return {
            "demographic_parity_diff": float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
            ),
            "equalized_odds_diff": float(
                equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
            ),
        }

    def is_biased(self, results: dict[str, float], threshold: float = 0.1) -> bool:
        dpd = abs(results.get("demographic_parity_diff", 0.0))
        eod = abs(results.get("equalized_odds_diff", 0.0))
        return dpd > threshold or eod > threshold

    def summary(self, results: dict[str, float]) -> str:
        dpd = results.get("demographic_parity_diff", 0.0)
        eod = results.get("equalized_odds_diff", 0.0)
        bias = "YES" if self.is_biased(results) else "NO"
        return f"Demographic Parity: {dpd:.4f}, Equalized Odds: {eod:.4f}, Bias: {bias}"
