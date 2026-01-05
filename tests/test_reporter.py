# tests for ReporterAgent
import pytest
from fairhire.agents.reporter import ReporterAgent

def test_generate_markdown():
  reporter = ReporterAgent()
  findings = [
    {"type": "Data Bias", "is_biased": True, "summary": "High disparity"},
    {"type": "Model Bias", "is_biased": False, "summary": "OK"},
  ]
  report = reporter.generate(findings)
  assert "# FairHire Audit Report" in report
  assert "[BIAS]" in report
  assert "[OK]" in report

def test_generate_json():
  reporter = ReporterAgent("Test Report")
  findings = [{"type": "Test", "is_biased": True}]
  result = reporter.generate_json(findings)
  assert result["title"] == "Test Report"
  assert result["total"] == 1
  assert result["bias_count"] == 1
