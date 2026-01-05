# tests for ModelBiasAgent
import numpy as np
import pytest
from fairhire.agents.model_bias import ModelBiasAgent

def test_compute_metrics():
  agent = ModelBiasAgent()
  y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
  y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])  # biased toward group 1
  sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
  result = agent.compute_metrics(y_true, y_pred, sensitive)
  assert "demographic_parity_diff" in result
  assert "equalized_odds_diff" in result

def test_is_biased():
  agent = ModelBiasAgent()
  result = {"demographic_parity_diff": 0.25, "equalized_odds_diff": 0.1}
  assert agent.is_biased(result, threshold=0.1) == True
  assert agent.is_biased(result, threshold=0.3) == False

def test_summary():
  agent = ModelBiasAgent()
  result = {"demographic_parity_diff": 0.25, "equalized_odds_diff": 0.1}
  summary = agent.summary(result)
  assert "Demographic Parity" in summary
  assert "Equalized Odds" in summary
