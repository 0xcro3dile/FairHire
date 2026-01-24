# L20 compliant SHAP-based explainer agent
# Strict types, minimal deps, structured explanations
from collections.abc import Callable
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class ExplainerAgent:
    """
    SHAP-based model explanation agent.
    
    L20: Uses KernelSHAP for model-agnostic explanations.
    Falls back to permutation importance if SHAP unavailable.
    """
    
    def __init__(self, background_data: np.ndarray, feature_names: list[str]) -> None:
        self.background = background_data
        self.feature_names = feature_names
        self._shap_available = self._check_shap()
    
    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            log.warning("shap_not_available", fallback="permutation_importance")
            return False
    
    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> dict[str, Any]:
        """
        Generate explanation for a single instance.
        
        Returns dict with feature contributions and prediction.
        """
        if self._shap_available:
            return self._shap_explain(predict_fn, instance)
        return self._permutation_explain(predict_fn, instance)
    
    def _shap_explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> dict[str, Any]:
        """SHAP-based explanation using KernelSHAP."""
        import shap
        
        # Sample background for efficiency (max 100)
        bg_sample = self.background
        if len(self.background) > 100:
            indices = np.random.choice(len(self.background), 100, replace=False)
            bg_sample = self.background[indices]
        
        explainer = shap.KernelExplainer(predict_fn, bg_sample)
        shap_values = explainer.shap_values(instance.reshape(1, -1))[0]
        
        # Build contribution dict
        contributions: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(shap_values[i])
        
        # Sort by absolute importance
        sorted_contribs = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        prediction = float(predict_fn(instance.reshape(1, -1))[0])
        
        return {
            "method": "shap",
            "prediction": prediction,
            "base_value": float(explainer.expected_value),
            "contributions": sorted_contribs,
            "top_features": list(sorted_contribs.keys())[:5],
        }
    
    def _permutation_explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> dict[str, Any]:
        """Fallback permutation-based importance."""
        base_pred = float(predict_fn(instance.reshape(1, -1))[0])
        contributions: dict[str, float] = {}
        
        for i, name in enumerate(self.feature_names):
            # Permute single feature using background mean
            perturbed = instance.copy()
            perturbed[i] = float(np.mean(self.background[:, i]))
            perturbed_pred = float(predict_fn(perturbed.reshape(1, -1))[0])
            contributions[name] = base_pred - perturbed_pred
        
        sorted_contribs = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return {
            "method": "permutation",
            "prediction": base_pred,
            "base_value": float(np.mean(predict_fn(self.background))),
            "contributions": sorted_contribs,
            "top_features": list(sorted_contribs.keys())[:5],
        }
