# Generates markdown/json audit reports
# L20 compliant: strict type hints
from datetime import datetime
from typing import Any


class ReporterAgent:
  def __init__(self, title: str = "FairHire Audit Report") -> None:
    self.title = title

  def generate(self, findings: list[dict[str, Any]]) -> str:
    bias_count = sum(1 for f in findings if f.get("is_biased"))
    lines = [
      f"# {self.title}",
      f"Generated: {datetime.now().isoformat()}",
      "",
      f"## Summary\nTotal: {len(findings)}, Bias Detected: {bias_count}",
      "",
      "## Findings",
      "",
    ]
    for i, f in enumerate(findings, 1):
      status = "[BIAS]" if f.get("is_biased") else "[OK]"
      finding_type = f.get("type", "Check")
      lines.append(f"### {i}. {finding_type} {status}")
      summary = f.get("summary")
      if summary:
        lines.append(summary)
      lines.append("")
    return "\n".join(lines)

  def generate_json(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
    bias_count = sum(1 for f in findings if f.get("is_biased"))
    return {
      "title": self.title,
      "generated": datetime.now().isoformat(),
      "total": len(findings),
      "bias_count": bias_count,
      "findings": findings,
    }
