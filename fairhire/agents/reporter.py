# generates markdown/json audit reports
from datetime import datetime

class ReporterAgent:
  def __init__(self, title: str = "FairHire Audit Report"): self.title = title

  def generate(self, findings: list[dict]) -> str:
    bias_count = sum(1 for f in findings if f.get("is_biased"))
    lines = [f"# {self.title}", f"Generated: {datetime.now().isoformat()}", "",
             f"## Summary\nTotal: {len(findings)}, Bias Detected: {bias_count}", "", "## Findings", ""]
    for i, f in enumerate(findings, 1):
      status = "[BIAS]" if f.get("is_biased") else "[OK]"
      lines.append(f"### {i}. {f.get('type', 'Check')} {status}")
      if "summary" in f: lines.append(f.get("summary"))
      lines.append("")
    return "\n".join(lines)

  def generate_json(self, findings: list[dict]) -> dict:
    return {"title": self.title, "generated": datetime.now().isoformat(),
            "total": len(findings), "bias_count": sum(1 for f in findings if f.get("is_biased")), "findings": findings}
