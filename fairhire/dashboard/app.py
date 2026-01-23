# simple streamlit dashboard for bias audits
import streamlit as st
import pandas as pd
import tempfile, os

st.set_page_config(page_title="FairHire Auditor", page_icon="⚖️", layout="wide")
st.title("FairHire Auditor")
st.markdown("Upload hiring data to detect bias")

# file upload
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded, encoding="utf-8")
        if df.empty:
            st.error("CSV file is empty. Please upload a file with data.")
            st.stop()
    except pd.errors.EmptyDataError:
        st.error("CSV file is empty or corrupted.")
        st.stop()
    except UnicodeDecodeError:
        st.error("CSV encoding not supported. Please save as UTF-8.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read CSV: {str(e)}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(5))

    # dynamic config based on df columns
    st.sidebar.header("Configuration")

    # smart defaults
    default_prot = "gender" if "gender" in df.columns else df.columns[0]
    protected_attr = st.sidebar.selectbox(
        "Protected Attribute", df.columns, index=list(df.columns).index(default_prot)
    )

    default_label = "hired" if "hired" in df.columns else df.columns[-1]
    label_col = st.sidebar.selectbox(
        "Label Column", df.columns, index=list(df.columns).index(default_label)
    )

    # dynamic value selection (handles strings/ints correctly)
    unique_vals = list(df[protected_attr].unique())
    priv_value = st.sidebar.selectbox("Privileged Value", unique_vals)
    unpriv_value = st.sidebar.selectbox(
        "Unprivileged Value", [v for v in unique_vals if v != priv_value]
    )

    if st.button("Run Audit", disabled=st.session_state.get("running", False)):
        st.session_state["running"] = True
        with st.spinner("Analyzing..."):
            # save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            try:
                from fairhire.core.orchestrator import Orchestrator

                orch = Orchestrator()
                result = orch.run_audit(
                    tmp_path,
                    [protected_attr],
                    [{protected_attr: priv_value}],
                    [{protected_attr: unpriv_value}],
                    label_col,
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
            except Exception as e:
                st.error(f"Audit Failed: {str(e)}")
            finally:
                os.unlink(tmp_path)
                st.session_state["running"] = False
else:
    st.info("Upload a CSV to start configuration")
