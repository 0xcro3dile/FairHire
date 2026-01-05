# simple streamlit dashboard for bias audits
import streamlit as st
import pandas as pd
import tempfile, os

st.set_page_config(page_title="FairHire Auditor", page_icon="⚖️", layout="wide")
st.title("FairHire Auditor")
st.markdown("Upload hiring data to detect bias")

# sidebar config
st.sidebar.header("Configuration")
protected_attr = st.sidebar.selectbox("Protected Attribute", ["gender", "race", "age"])
label_col = st.sidebar.text_input("Label Column", "hired")
priv_value = st.sidebar.number_input("Privileged Value", value=1, step=1)
unpriv_value = st.sidebar.number_input("Unprivileged Value", value=0, step=1)

# file upload
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
  df = pd.read_csv(uploaded)
  st.subheader("Data Preview")
  st.dataframe(df.head(10))

  if st.button("Run Audit"):
    with st.spinner("Analyzing..."):
      # save to temp file
      with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
      try:
        from fairhire.core.orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch.run_audit(
          tmp_path, [protected_attr],
          [{protected_attr: priv_value}], [{protected_attr: unpriv_value}],
          label_col
        )
        st.success("Audit Complete!")
        st.subheader("Findings")
        for finding in result["findings"]:
          status = "🔴" if finding["is_biased"] else "🟢"
          st.markdown(f"### {status} {finding['type']}")
          st.write(finding["summary"])
          if "metrics" in finding:
            cols = st.columns(len(finding["metrics"]))
            for i, (k, v) in enumerate(finding["metrics"].items()):
              cols[i].metric(k, f"{v:.4f}" if isinstance(v, float) else v)
        st.subheader("Report")
        st.markdown(result["report"])
      finally:
        os.unlink(tmp_path)
else:
  st.info("Upload a CSV to start")
