# LangGraph orchestrator - coordinates bias detection agents
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd

from fairhire.agents.data_bias import DataBiasAgent
from fairhire.agents.model_bias import ModelBiasAgent
from fairhire.agents.explainer import ExplainerAgent
from fairhire.agents.reporter import ReporterAgent

class AuditState(TypedDict):  # shared state passed between nodes
  dataset_path: str
  protected_attrs: list[str]
  privileged_groups: list[dict]
  unprivileged_groups: list[dict]
  label_col: str
  model_predict_fn: object  # optional: model prediction function
  findings: list[dict]
  explanations: list[dict]
  report: str
  status: str

class Orchestrator:
  def __init__(self):
    self.graph = StateGraph(AuditState)
    self.graph.add_node("data_bias", self._run_data_bias)
    self.graph.add_node("model_bias", self._run_model_bias)
    self.graph.add_node("explainer", self._run_explainer)
    self.graph.add_node("reporter", self._run_reporter)
    # flow: START -> data_bias -> model_bias -> explainer -> reporter -> END
    self.graph.add_edge(START, "data_bias")
    self.graph.add_edge("data_bias", "model_bias")
    self.graph.add_edge("model_bias", "explainer")
    self.graph.add_edge("explainer", "reporter")
    self.graph.add_edge("reporter", END)
    self.compiled = self.graph.compile()

  def _run_data_bias(self, state: AuditState) -> dict:
    df = pd.read_csv(state["dataset_path"])
    agent = DataBiasAgent(state["protected_attrs"], state["privileged_groups"], state["unprivileged_groups"])
    results = agent.analyze(df, state["label_col"])
    finding = {"type": "Data Bias", "is_biased": agent.is_biased(results), "summary": agent.summary(results), "metrics": results}
    return {"findings": state["findings"] + [finding], "status": "data_bias_complete"}

  def _run_model_bias(self, state: AuditState) -> dict:
    if not state.get("model_predict_fn"): return {"status": "model_bias_skipped"}  # no model provided
    df = pd.read_csv(state["dataset_path"])
    y_true = df[state["label_col"]].values
    sensitive = df[state["protected_attrs"][0]].values
    y_pred = state["model_predict_fn"](df.drop(columns=[state["label_col"]]).values)
    agent = ModelBiasAgent()
    results = agent.compute_metrics(y_true, y_pred, sensitive)
    finding = {"type": "Model Bias", "is_biased": agent.is_biased(results), "summary": agent.summary(results), "metrics": results}
    return {"findings": state["findings"] + [finding], "status": "model_bias_complete"}

  def _run_explainer(self, state: AuditState) -> dict:
    if not state.get("model_predict_fn"): return {"explanations": [], "status": "explainer_skipped"}
    df = pd.read_csv(state["dataset_path"])
    X = df.drop(columns=[state["label_col"]]).values
    agent = ExplainerAgent(X, list(df.drop(columns=[state["label_col"]]).columns))
    # explain first 3 instances
    explanations = [agent.explain(state["model_predict_fn"], X[i]) for i in range(min(3, len(X)))]
    return {"explanations": explanations, "status": "explainer_complete"}

  def _run_reporter(self, state: AuditState) -> dict:
    report = ReporterAgent().generate(state["findings"])
    return {"report": report, "status": "complete"}

  def run_audit(self, dataset_path: str, protected_attrs: list[str], privileged_groups: list[dict],
                unprivileged_groups: list[dict], label_col: str = "hired", model_predict_fn=None) -> dict:
    return self.compiled.invoke({
      "dataset_path": dataset_path, "protected_attrs": protected_attrs,
      "privileged_groups": privileged_groups, "unprivileged_groups": unprivileged_groups,
      "label_col": label_col, "model_predict_fn": model_predict_fn,
      "findings": [], "explanations": [], "report": "", "status": "pending"
    })
