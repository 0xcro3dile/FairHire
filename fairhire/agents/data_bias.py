# AIF360 wrapper for statistical bias detection
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

class DataBiasAgent:
  def __init__(self, protected_attrs: list[str], privileged_groups: list[dict], unprivileged_groups: list[dict]):
    self.protected_attrs = protected_attrs
    self.privileged_groups = privileged_groups
    self.unprivileged_groups = unprivileged_groups

  def analyze(self, df: pd.DataFrame, label_col: str, favorable_label: float = 1.0) -> dict:
    # convert pandas df to AIF360 dataset format
    dataset = BinaryLabelDataset(
      df=df, label_names=[label_col], protected_attribute_names=self.protected_attrs,
      favorable_label=favorable_label, unfavorable_label=0.0
    )
    metric = BinaryLabelDatasetMetric(dataset, self.unprivileged_groups, self.privileged_groups)
    return {
      "statistical_parity": metric.statistical_parity_difference(),  # Pr(Y=1|unprivileged) - Pr(Y=1|privileged)
      "disparate_impact": metric.disparate_impact(),  # ratio of base rates
      "base_rate_privileged": metric.base_rate(privileged=True),
      "base_rate_unprivileged": metric.base_rate(privileged=False),
      "num_positives": metric.num_positives(),
      "num_negatives": metric.num_negatives(),
    }

  def is_biased(self, results: dict, threshold: float = 0.1) -> bool:
    return abs(results["statistical_parity"]) > threshold

  def summary(self, results: dict) -> str:
    sp, di = results["statistical_parity"], results["disparate_impact"]
    return f"Statistical Parity: {sp:.4f}, Disparate Impact: {di:.4f}, Bias: {'YES' if self.is_biased(results) else 'NO'}"
