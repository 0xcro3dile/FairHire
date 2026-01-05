# Fairlearn wrapper for model prediction bias
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

class ModelBiasAgent:
  def __init__(self, sensitive_features: list[str] = None):
    self.sensitive_features = sensitive_features or []

  def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict:
    return {
      "demographic_parity_diff": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
      "equalized_odds_diff": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
    }

  def is_biased(self, results: dict, threshold: float = 0.1) -> bool:
    return abs(results["demographic_parity_diff"]) > threshold

  def summary(self, results: dict) -> str:
    dpd, eod = results["demographic_parity_diff"], results["equalized_odds_diff"]
    return f"Demographic Parity: {dpd:.4f}, Equalized Odds: {eod:.4f}, Bias: {'YES' if self.is_biased(results) else 'NO'}"
