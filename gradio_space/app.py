"""
Gradio app (Hugging Face Space): one tab per model.
Loads the same saved artifacts as api.py via inference.py (no retraining).
"""
import json
import sys
from pathlib import Path

import gradio as gr

# Repo layout: inference.py is one folder up. In a Hugging Face Space it sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference as inf  # noqa: E402

SPECS = inf.get_specs()


def _components(fields):
    comps = []
    for f in fields:
        label = inf._label(f["name"])
        if f["kind"] == "number":
            comps.append(gr.Number(label=label, value=f["default"]))
        elif f["kind"] == "choice":
            choices = [str(o) for o in f["options"]]
            comps.append(gr.Dropdown(choices=choices, value=str(f["default"]), label=label))
        else:
            comps.append(gr.Textbox(label=label, value=f["default"]))
    return comps


def _summary(task, result):
    if task == "regression":
        return f"### Predicted daily sales revenue: **KES {result['predicted_daily_sales_revenue_kes']:,.2f}**"
    if task == "classification":
        return f"### Predicted category: **{result['predicted_category']}**"
    if task == "clustering":
        return f"### Cluster {result['predicted_cluster']}: **{result['segment_name']}**"
    recs = result["recommendations"]
    if not recs:
        return "### " + result.get("message", "No recommendations.")
    lines = [f"{i}. **{r['show_name']}** (confidence {r['confidence']}, lift {r['lift']})"
             for i, r in enumerate(recs, 1)]
    return "### Recommended for you\n" + "\n".join(lines)


def _make_handler(task, names, fn):
    def handler(*values):
        try:
            result = fn(dict(zip(names, values)))
        except ValueError as e:
            raise gr.Error(str(e))
        return _summary(task, result), json.dumps(result, indent=2, default=str)
    return handler


def _model_tab(task, title, description, fn):
    fields = SPECS[task]
    gr.Markdown(description)
    inputs = _components(fields)
    button = gr.Button(f"Predict ({title.lower()})", variant="primary")
    headline = gr.Markdown()
    raw = gr.Code(label="Raw response", language="json")
    names = [f["name"] for f in fields]
    button.click(_make_handler(task, names, fn), inputs=inputs, outputs=[headline, raw])


def _recommend_handler(items, top_n):
    try:
        result = inf.recommend_association(items, top_n)
    except ValueError as e:
        raise gr.Error(str(e))
    return _summary("recommender", result), json.dumps(result, indent=2, default=str)


with gr.Blocks(title="ML Models Demo") as demo:
    gr.Markdown(
        "# ML Models Demo\n"
        "Four models trained by our group, loaded from saved files "
        "(the same artifacts served by our Flask API)."
    )
    with gr.Tabs():
        with gr.Tab("Regression"):
            _model_tab("regression", "Regression",
                       "Predict a retail store's **daily sales revenue (KES)**.",
                       inf.predict_regression)
        with gr.Tab("Classification"):
            _model_tab("classification", "Classification",
                       "Classify a company's **financial distress** category.",
                       inf.predict_classification)
        with gr.Tab("Clustering"):
            _model_tab("clustering", "Clustering",
                       "Find which **customer segment** (k-Means cluster) a client belongs to.",
                       inf.predict_clustering)
        with gr.Tab("Recommender"):
            gr.Markdown("Pick shows you have watched and get **association-rule recommendations**.")
            shows = gr.Dropdown(choices=SPECS["shows"], multiselect=True,
                                value=SPECS["shows"][:2], label="Shows watched")
            top_n = gr.Slider(1, 10, value=5, step=1, label="Number of recommendations")
            btn = gr.Button("Recommend", variant="primary")
            headline = gr.Markdown()
            raw = gr.Code(label="Raw response", language="json")
            btn.click(_recommend_handler, inputs=[shows, top_n], outputs=[headline, raw])

if __name__ == "__main__":
    demo.launch()
