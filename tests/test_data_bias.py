# tests for DataBiasAgent
import pandas as pd
import pytest
from fairhire.agents.data_bias import DataBiasAgent

def sample_df():
  return pd.DataFrame({
    "gender": [1, 1, 1, 1, 0, 0, 0, 0],
    "hired": [1, 1, 1, 0, 0, 0, 1, 0],
  })

def test_analyze_returns_metrics():
  agent = DataBiasAgent(["gender"], [{"gender": 1}], [{"gender": 0}])
  result = agent.analyze(sample_df(), "hired")
  assert "statistical_parity" in result
  assert "disparate_impact" in result
  assert isinstance(result["statistical_parity"], float)

def test_is_biased_detects_bias():
  agent = DataBiasAgent(["gender"], [{"gender": 1}], [{"gender": 0}])
  result = agent.analyze(sample_df(), "hired")
  # sample data has 75% hire rate for men vs 25% for women = 0.5 diff
  assert agent.is_biased(result, threshold=0.1) == True

def test_summary_format():
  agent = DataBiasAgent(["gender"], [{"gender": 1}], [{"gender": 0}])
  result = agent.analyze(sample_df(), "hired")
  summary = agent.summary(result)
  assert "Statistical Parity" in summary
  assert "Disparate Impact" in summary
