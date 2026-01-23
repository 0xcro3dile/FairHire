# LangGraph orchestrator - coordinates bias detection agents
# L20 compliant: strict types, structlog, Callable over object
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import structlog
from langgraph.graph import END, START, StateGraph

from fairhire.agents.data_bias import DataBiasAgent
from fairhire.agents.explainer import ExplainerAgent
from fairhire.agents.model_bias import ModelBiasAgent
from fairhire.agents.reporter import ReporterAgent

log = structlog.get_logger(__name__)

# Type alias for model prediction function
PredictFn = Callable[[np.ndarray], np.ndarray]


class AuditState(TypedDict, total=False):
    """Shared state passed between nodes."""

    dataset_path: str
    protected_attrs: list[str]
    privileged_groups: list[dict[str, Any]]
    unprivileged_groups: list[dict[str, Any]]
    label_col: str
    model_predict_fn: PredictFn | None
    findings: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    report: str
    status: str
    df: pd.DataFrame | None


class Orchestrator:
    def __init__(self) -> None:
        self.graph: StateGraph[AuditState] = StateGraph(AuditState)
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

    def _run_data_bias(self, state: AuditState) -> dict[str, Any]:
        log.info("data_bias_start", dataset=state.get("dataset_path"))
        df = self._get_dataframe(state)
        self._validate_columns(df, state["protected_attrs"], state["label_col"])
        agent = DataBiasAgent(
            state["protected_attrs"],
            state["privileged_groups"],
            state["unprivileged_groups"],
        )
        results = agent.analyze(df, state["label_col"])
        finding = {
            "type": "Data Bias",
            "is_biased": agent.is_biased(results),
            "summary": agent.summary(results),
            "metrics": results,
        }
        log.info("data_bias_complete", biased=finding["is_biased"])
        return {
            "findings": state.get("findings", []) + [finding],
            "status": "data_bias_complete",
            "df": df,
        }

    def _run_model_bias(self, state: AuditState) -> dict[str, Any]:
        predict_fn = state.get("model_predict_fn")
        if not predict_fn:
            return {"status": "model_bias_skipped"}
        log.info("model_bias_start")
        df = self._get_dataframe(state)
        y_true = df[state["label_col"]].values
        sensitive = df[state["protected_attrs"][0]].values
        try:
            y_pred = predict_fn(df.drop(columns=[state["label_col"]]).values)
            if not isinstance(y_pred, np.ndarray) or y_pred.shape[0] != len(y_true):
                raise ValueError(
                    f"Model predictions invalid shape: expected ({len(y_true)},), "
                    f"got {y_pred.shape if isinstance(y_pred, np.ndarray) else type(y_pred)}"
                )
        except Exception as e:
            log.error("model_prediction_failed", error=str(e))
            return {"status": "model_bias_failed", "error": str(e)}
        agent = ModelBiasAgent()
        results = agent.compute_metrics(y_true, y_pred, sensitive)
        finding = {
            "type": "Model Bias",
            "is_biased": agent.is_biased(results),
            "summary": agent.summary(results),
            "metrics": results,
        }
        log.info("model_bias_complete", biased=finding["is_biased"])
        return {"findings": state.get("findings", []) + [finding], "status": "model_bias_complete"}

    def _run_explainer(self, state: AuditState) -> dict[str, Any]:
        predict_fn = state.get("model_predict_fn")
        if not predict_fn:
            return {"explanations": [], "status": "explainer_skipped"}
        log.info("explainer_start")
        df = self._get_dataframe(state)
        feature_cols = [c for c in df.columns if c != state["label_col"]]
        X = df[feature_cols].values
        indices = self._sample_diverse_indices(df, state["label_col"], n=3)
        agent = ExplainerAgent(X, feature_cols)
        explanations = [agent.explain(predict_fn, X[i]) for i in indices]
        log.info("explainer_complete", count=len(explanations))
        return {"explanations": explanations, "status": "explainer_complete"}

    def _run_reporter(self, state: AuditState) -> dict[str, Any]:
        log.info("reporter_start")
        report = ReporterAgent().generate(state.get("findings", []))
        log.info("audit_complete")
        return {"report": report, "status": "complete"}

    def _get_dataframe(self, state: AuditState) -> pd.DataFrame:
        """Cache dataframe in state to avoid reading CSV multiple times."""
        cached = state.get("df")
        if cached is not None:
            return cached
        log.debug("loading_csv", path=state.get("dataset_path"))
        return pd.read_csv(state["dataset_path"])

    def _validate_columns(
        self, df: pd.DataFrame, protected_attrs: list[str], label_col: str
    ) -> None:
        """Validate required columns exist in dataframe."""
        missing: list[str] = []
        if label_col not in df.columns:
            missing.append(label_col)
        for attr in protected_attrs:
            if attr not in df.columns:
                missing.append(attr)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    def _sample_diverse_indices(self, df: pd.DataFrame, label_col: str, n: int = 3) -> list[int]:
        """Sample diverse instances (stratified by label) for explanation."""
        if len(df) <= n:
            return list(range(len(df)))
        pos_idx = df[df[label_col] == 1].index.tolist()
        neg_idx = df[df[label_col] == 0].index.tolist()
        selected: list[int] = []
        if pos_idx:
            selected.append(pos_idx[0])
        if neg_idx:
            selected.append(neg_idx[0])
        remaining = [i for i in range(len(df)) if i not in selected]
        if remaining and len(selected) < n:
            import random

            selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))
        return selected[:n]

    def run_audit(
        self,
        dataset_path: str,
        protected_attrs: list[str],
        privileged_groups: list[dict[str, Any]],
        unprivileged_groups: list[dict[str, Any]],
        label_col: str = "hired",
        model_predict_fn: PredictFn | None = None,
    ) -> dict[str, Any]:
        initial_state: AuditState = {
            "dataset_path": dataset_path,
            "protected_attrs": protected_attrs,
            "privileged_groups": privileged_groups,
            "unprivileged_groups": unprivileged_groups,
            "label_col": label_col,
            "model_predict_fn": model_predict_fn,
            "findings": [],
            "explanations": [],
            "report": "",
            "status": "pending",
            "df": None,
        }
        return self.compiled.invoke(initial_state)
