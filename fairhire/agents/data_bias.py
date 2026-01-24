# AIF360 wrapper for statistical bias detection
# L20 compliant: strict type hints
from typing import Any

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


class DataBiasAgent:
    def __init__(
        self,
        protected_attrs: list[str],
        privileged_groups: list[dict[str, Any]],
        unprivileged_groups: list[dict[str, Any]],
    ) -> None:
        self.protected_attrs = protected_attrs
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

    def analyze(
        self, df: pd.DataFrame, label_col: str, favorable_label: float = 1.0
    ) -> dict[str, float | int]:
        # Convert pandas df to AIF360 dataset format
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[label_col],
            protected_attribute_names=self.protected_attrs,
            favorable_label=favorable_label,
            unfavorable_label=0.0,
        )
        metric = BinaryLabelDatasetMetric(dataset, self.unprivileged_groups, self.privileged_groups)
        return {
            "statistical_parity": metric.statistical_parity_difference(),
            "disparate_impact": metric.disparate_impact(),
            "base_rate_privileged": metric.base_rate(privileged=True),
            "base_rate_unprivileged": metric.base_rate(privileged=False),
            "num_positives": metric.num_positives(),
            "num_negatives": metric.num_negatives(),
        }

    def is_biased(self, results: dict[str, float | int], threshold: float = 0.1) -> bool:
        sp = results.get("statistical_parity", 0.0)
        return abs(float(sp)) > threshold

    def summary(self, results: dict[str, float | int]) -> str:
        sp = results.get("statistical_parity", 0.0)
        di = results.get("disparate_impact", 0.0)
        bias = "YES" if self.is_biased(results) else "NO"
        return f"Statistical Parity: {sp:.4f}, Disparate Impact: {di:.4f}, Bias: {bias}"
