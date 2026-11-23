"""
Streamlit app: one tab per model.
Loads the same saved artifacts as api.py via inference.py (no retraining).
"""
import json
import sys
from pathlib import Path

import streamlit as st

# inference.py lives one folder up (repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference as inf  # noqa: E402

st.set_page_config(page_title="ML Models Demo", layout="centered")
st.title("ML Models Demo")
st.caption("Four models trained by our group, loaded from saved files "
           "(the same artifacts served by our Flask API).")

SPECS = inf.get_specs()


def _widget(field, key):
    label = inf._label(field["name"])
    if field["kind"] == "number":
        return st.number_input(label, value=float(field["default"]), format="%g", key=key)
    if field["kind"] == "choice":
        options = list(field["options"])
        index = options.index(field["default"]) if field["default"] in options else 0
        return st.selectbox(label, options, index=index, key=key)
    return st.text_input(label, value=field["default"], key=key)


def _show(task, result):
    if task == "regression":
        st.success(f"Predicted daily sales revenue: KES {result['predicted_daily_sales_revenue_kes']:,.2f}")
    elif task == "classification":
        st.success(f"Predicted category: {result['predicted_category']}")
        if "class_probabilities" in result:
            st.bar_chart(result["class_probabilities"])
    elif task == "clustering":
        st.success(f"Cluster {result['predicted_cluster']}: {result['segment_name']}")
    else:
        recs = result["recommendations"]
        if not recs:
            st.info(result.get("message", "No recommendations."))
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. **{r['show_name']}**  (confidence {r['confidence']}, lift {r['lift']})")
    with st.expander("Raw response"):
        st.code(json.dumps(result, indent=2, default=str), language="json")


def _model_tab(task, description, fn):
    st.write(description)
    with st.form(f"form_{task}"):
        values = {f["name"]: _widget(f, f"{task}_{f['name']}") for f in SPECS[task]}
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        try:
            _show(task, fn(values))
        except ValueError as e:
            st.error(str(e))


tab_reg, tab_cls, tab_clu, tab_rec = st.tabs(
    ["Regression", "Classification", "Clustering", "Recommender"]
)

with tab_reg:
    _model_tab("regression", "Predict a retail store's **daily sales revenue (KES)**.",
               inf.predict_regression)
with tab_cls:
    _model_tab("classification", "Classify a company's **financial distress** category.",
               inf.predict_classification)
with tab_clu:
    _model_tab("clustering", "Find which **customer segment** (k-Means cluster) a client belongs to.",
               inf.predict_clustering)
with tab_rec:
    st.write("Pick shows you have watched and get **association-rule recommendations**.")
    with st.form("form_recommender"):
        shows = st.multiselect("Shows watched", SPECS["shows"], default=SPECS["shows"][:2])
        top_n = st.slider("Number of recommendations", 1, 10, 5)
        submitted = st.form_submit_button("Recommend", type="primary")
    if submitted:
        try:
            _show("recommender", inf.recommend_association(shows, top_n))
        except ValueError as e:
            st.error(str(e))
