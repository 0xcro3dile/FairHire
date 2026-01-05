# LangGraph orchestrator - coordinates bias detection agents
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd

from fairhire.agents.data_bias import DataBiasAgent
from fairhire.agents.reporter import ReporterAgent

class AuditState(TypedDict):  # shared state passed between nodes
  dataset_path: str
  protected_attrs: list[str]
  privileged_groups: list[dict]
  unprivileged_groups: list[dict]
  label_col: str
  findings: list[dict]
  report: str
  status: str

class Orchestrator:
  def __init__(self):
    self.graph = StateGraph(AuditState)
    self.graph.add_node("data_bias", self._run_data_bias)
    self.graph.add_node("reporter", self._run_reporter)
    # flow: START -> data_bias -> reporter -> END
    self.graph.add_edge(START, "data_bias")
    self.graph.add_edge("data_bias", "reporter")
    self.graph.add_edge("reporter", END)
    self.compiled = self.graph.compile()

  def _run_data_bias(self, state: AuditState) -> dict:
    df = pd.read_csv(state["dataset_path"])
    agent = DataBiasAgent(state["protected_attrs"], state["privileged_groups"], state["unprivileged_groups"])
    results = agent.analyze(df, state["label_col"])
    finding = {"type": "Data Bias", "is_biased": agent.is_biased(results), "summary": agent.summary(results), "metrics": results}
    return {"findings": state["findings"] + [finding], "status": "data_bias_complete"}

  def _run_reporter(self, state: AuditState) -> dict:
    report = ReporterAgent().generate(state["findings"])
    return {"report": report, "status": "complete"}

  def run_audit(self, dataset_path: str, protected_attrs: list[str], privileged_groups: list[dict],
                unprivileged_groups: list[dict], label_col: str = "hired") -> dict:
    return self.compiled.invoke({
      "dataset_path": dataset_path, "protected_attrs": protected_attrs,
      "privileged_groups": privileged_groups, "unprivileged_groups": unprivileged_groups,
      "label_col": label_col, "findings": [], "report": "", "status": "pending"
    })
