# LIME wrapper for explaining individual predictions
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

class ExplainerAgent:
  def __init__(self, training_data: np.ndarray, feature_names: list[str], class_names: list[str] = None):
    self.explainer = LimeTabularExplainer(
      training_data, feature_names=feature_names, class_names=class_names or ["Reject", "Hire"], mode="classification"
    )
    self.feature_names = feature_names

  def explain(self, predict_fn, instance: np.ndarray, num_features: int = 5) -> dict:
    exp = self.explainer.explain_instance(instance, predict_fn, num_features=num_features)
    return {"features": exp.as_list(), "score": exp.score, "prediction": exp.predict_proba.tolist()}

  def explain_text(self, predict_fn, instance: np.ndarray, num_features: int = 5) -> str:
    result = self.explain(predict_fn, instance, num_features)
    lines = ["Key factors:"]
    for feat, weight in result["features"]:
      lines.append(f"  • {feat}: {'↑' if weight > 0 else '↓'} ({weight:.3f})")
    return "\n".join(lines)
