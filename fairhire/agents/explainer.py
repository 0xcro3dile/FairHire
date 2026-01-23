# LIME wrapper for explaining individual predictions
# L20 compliant: strict type hints, Callable
from collections.abc import Callable

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from numpy.typing import NDArray


class ExplainerAgent:
    def __init__(
        self,
        training_data: NDArray[np.float64],
        feature_names: list[str],
        class_names: list[str] | None = None,
    ) -> None:
        self.explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names or ["Reject", "Hire"],
            mode="classification",
        )
        self.feature_names = feature_names

    def explain(
        self,
        predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        instance: NDArray[np.float64],
        num_features: int = 5,
    ) -> dict[str, list[tuple[str, float]] | float | list[float]]:
        exp = self.explainer.explain_instance(instance, predict_fn, num_features=num_features)
        return {
            "features": exp.as_list(),
            "score": exp.score,
            "prediction": exp.predict_proba.tolist(),
        }

    def explain_text(
        self,
        predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        instance: NDArray[np.float64],
        num_features: int = 5,
    ) -> str:
        result = self.explain(predict_fn, instance, num_features)
        lines = ["Key factors:"]
        features = result.get("features", [])
        if isinstance(features, list):
            for feat, weight in features:
                arrow = "↑" if weight > 0 else "↓"
                lines.append(f"  • {feat}: {arrow} ({weight:.3f})")
        return "\n".join(lines)
