"""
CardioScore Pro - Streamlit application.

Run:   streamlit run app.py
(Train the models first with:  python train_model.py)

CORE REQUIREMENT 5 : prediction interface (enter details -> result -> explanation)
EXTRA FEATURES     : A premium dashboard, C login/signup, D history + PDF/CSV,
                     E explainable AI, B per-model results.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth
import reporting
import train_model as tm

BASE_DIR = Path(__file__).resolve().parent
PAGES = ["Dashboard", "New Assessment", "History & Reports",
         "Model Performance", "Explainability", "About"]
RISK_COLORS = reporting.RISK_COLORS
LEVEL_MEANING = {
    "LOW": "Few of the entered values resemble patients with heart disease in the training data.",
    "MODERATE": "Some entered values resemble patients with heart disease and some do not.",
    "HIGH": "Many of the entered values resemble patients with heart disease in the training data.",
}
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-score", "roc_auc": "ROC-AUC"}

st.set_page_config(page_title="CardioScore Pro", page_icon="❤️", layout="wide",
                   initial_sidebar_state="expanded")


# ------------------------------------------------------------------ helpers
def load_css() -> None:
    css_path = BASE_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_artifacts(stamp: float):
    """Load metadata + models. `stamp` (file time) refreshes the cache after re-training."""
    return tm.load_artifacts()


def require_models():
    if not tm.artifacts_exist():
        st.markdown('<div class="cs-hero"><h1>CardioScore Pro</h1>'
                    '<p>The prediction models have not been trained yet.</p></div>',
                    unsafe_allow_html=True)
        st.error("Trained model files were not found in the models/ folder.")
        st.markdown("**Fix it in two steps:**")
        st.markdown("1. Put the UCI Cleveland dataset (`processed.cleveland.data`) in the "
                    "`data/` folder (see `data/README.md`) or stay online so it can be downloaded.")
        st.markdown("2. In a terminal, inside the project folder, run:")
        st.code("python train_model.py", language="bash")
        st.markdown("Then refresh this page.")
        st.stop()
    return get_artifacts((tm.MODELS_DIR / tm.METADATA_FILE).stat().st_mtime)


def esc(text) -> str:
    return html.escape(str(text))


def card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    cls = "cs-card accent" if accent else "cs-card"
    return (f'<div class="{cls}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def pill(level: str) -> str:
    return f'<span class="cs-pill {esc(level)}">{esc(level)} RISK</span>'


def disclaimer_box() -> None:
    st.markdown(f'<div class="cs-disclaimer"><b>Not a medical diagnosis.</b> '
                f'{esc(tm.DISCLAIMER)}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="cs-section-title">{esc(title)}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 320):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2937"),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def fmt_dt(iso_text: str) -> str:
    return reporting.format_datetime(iso_text)


def plot_time(iso_text: str) -> str:
    """Timestamp string that Plotly reads as a date (keeps same-minute points apart)."""
    try:
        return datetime.fromisoformat(iso_text).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(iso_text)


# ------------------------------------------------------------------- charts
def fig_gauge(prob: float, level: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 46}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[level]},
               "steps": [{"range": [0, 33], "color": "#dcfce7"},
                         {"range": [33, 66], "color": "#fef3c7"},
                         {"range": [66, 100], "color": "#fee2e2"}]}))
    return style_fig(fig, 260)


def fig_factors(df: pd.DataFrame):
    top = df.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["influence"], y=[f"{l}: {v}" for l, v in zip(top["label"], top["value"])],
        orientation="h",
        marker_color=["#b91c1c" if x > 0 else "#15803d" for x in top["influence"]],
        hovertemplate="%{y}<br>%{x:+.1f} percentage points<extra></extra>"))
    fig.update_layout(title="Effect of each value vs a typical patient",
                      xaxis_title="Change in risk (percentage points)")
    return style_fig(fig, 360)


def fig_models(probs: dict, selected: str):
    names = list(probs)
    fig = go.Figure(go.Bar(
        x=names, y=[probs[n] * 100 for n in names],
        marker_color=["#c0243b" if n == selected else "#94a3b8" for n in names],
        text=[f"{probs[n] * 100:.0f}%" for n in names], textposition="outside"))
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b")
    fig.update_layout(title="Same patient, every model", yaxis_title="Estimated probability (%)",
                      yaxis_range=[0, 110])
    return style_fig(fig, 320)


def fig_history(rows: list):
    ordered = list(reversed(rows))
    fig = go.Figure(go.Scatter(
        x=[plot_time(a["created_at"]) for a in ordered],
        y=[a["probability"] * 100 for a in ordered], mode="lines+markers",
        line=dict(color="#c0243b", width=3),
        marker=dict(size=10, color=[RISK_COLORS[a["risk_level"]] for a in ordered]),
        hovertemplate="%{x}<br>%{y:.0f}%<extra></extra>"))
    fig.update_layout(title="Estimated risk over time", yaxis_title="Probability (%)",
                      yaxis_range=[0, 100])
    return style_fig(fig, 320)


def fig_levels(rows: list):
    counts = {lvl: sum(1 for a in rows if a["risk_level"] == lvl)
              for lvl in ("LOW", "MODERATE", "HIGH")}
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.6,
                           marker=dict(colors=[RISK_COLORS[k] for k in counts]),
                           sort=False))
    fig.update_layout(title="Risk band split")
    return style_fig(fig, 320)


def fig_roc(meta: dict):
    fig = go.Figure()
    for name in meta["selection"]["ranking"]:
        m = meta["models"][name]
        fig.add_trace(go.Scatter(
            x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
            name=f"{name} (AUC {m['test']['roc_auc']:.2f})",
            line=dict(width=4 if name == meta["best_model"] else 2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(dash="dot", color="#94a3b8")))
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return style_fig(fig, 420)


def fig_metric_bars(meta: dict, metric: str):
    names = meta["selection"]["ranking"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Validation (5-fold CV mean)", x=names,
                         y=[meta["models"][n]["cv"][metric]["mean"] for n in names],
                         error_y=dict(type="data", array=[meta["models"][n]["cv"][metric]["std"]
                                                          for n in names]),
                         marker_color="#1d2f57"))
    fig.add_trace(go.Bar(name="Test set", x=names,
                         y=[meta["models"][n]["test"][metric] for n in names],
                         marker_color="#c0243b"))
    fig.update_layout(barmode="group", title=f"{METRIC_LABELS[metric]} by model",
                      yaxis_range=[0, 1.05])
    return style_fig(fig, 360)


def fig_confusion(cm: list):
    labels_x = ["Predicted: no disease", "Predicted: disease"]
    labels_y = ["Actual: no disease", "Actual: disease"]
    text = [[f"TN = {cm[0][0]}", f"FP = {cm[0][1]}"], [f"FN = {cm[1][0]}", f"TP = {cm[1][1]}"]]
    fig = go.Figure(go.Heatmap(z=cm, x=labels_x, y=labels_y, text=text,
                               texttemplate="%{text}", colorscale="Blues", showscale=False,
                               textfont={"size": 16}))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Confusion matrix (test set)")
    return style_fig(fig, 340)


def fig_importance(importance: dict, top: int = 13):
    items = list(importance.items())[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[v["mean"] for _, v in items],
        y=[tm.FEATURE_BY_NAME[k]["label"] for k, _ in items], orientation="h",
        error_x=dict(type="data", array=[v["std"] for _, v in items]),
        marker_color="#c0243b"))
    fig.update_layout(title="Global feature importance (permutation, test set)",
                      xaxis_title="Drop in ROC-AUC when the feature is shuffled")
    return style_fig(fig, 460)


# --------------------------------------------------------------- auth screen
def render_auth() -> None:
    """Render the branded login and account-creation screen."""
    st.markdown(
        """
        <div style="text-align:center; max-width:760px; margin:1.6rem auto 1.8rem; padding:1.2rem 1rem;">
            <div style="width:68px;height:68px;display:flex;align-items:center;justify-content:center;
                        margin:0 auto 1rem;border-radius:20px;color:white;font-size:2rem;
                        background:linear-gradient(145deg,#ed536b,#c92e50);
                        box-shadow:0 10px 28px rgba(201,46,80,.22);">♥</div>
            <h1 style="margin:0;font-size:clamp(2rem,4vw,3rem);font-weight:800;letter-spacing:-.04em;">
                CardioScore Pro
            </h1>
            <p style="margin:.55rem 0;font-size:.76rem;font-weight:750;letter-spacing:.13em;color:#c92e50;">
                HEART DISEASE RISK PREDICTION
            </p>
            <p style="margin:.9rem 0 .3rem;color:#64748b;font-size:.92rem;">
                Developed by <strong style="color:#1d2f57;">Shashant Sahu</strong>
            </p>
            <p style="max-width:560px;margin:.8rem auto 0;color:#64748b;line-height:1.7;">
                Machine learning-powered heart risk estimation with private accounts,
                assessment history and explainable results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, middle, _ = st.columns([1, 1.6, 1])
    with middle:
        tab_login, tab_signup = st.tabs(["Log in", "Create account"])
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                user = auth.authenticate_email(email, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Email or password is incorrect. If you have an older username account, enter that username in the email field.")

        with tab_signup:
            with st.form("signup_form"):
                full_name = st.text_input("Full name")
                new_email = st.text_input("Email address", placeholder="you@example.com")
                new_password = st.text_input(
                    "Create password", type="password",
                    help="At least 8 characters, including a letter and a number."
                )
                confirm = st.text_input("Confirm password", type="password")
                created = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if created:
                if new_password != confirm:
                    st.error("The passwords do not match.")
                else:
                    try:
                        st.session_state["user"] = auth.create_email_user(new_email, new_password, full_name)
                        st.rerun()
                    except auth.AuthError as exc:
                        st.error(str(exc))
        disclaimer_box()



def go_to(page: str) -> None:
    st.session_state["nav"] = page


def render_sidebar(user: dict) -> str:
    with st.sidebar:
        st.markdown('<div class="cs-brand"><div class="logo">❤️</div><div>'
                    '<div class="name">CardioScore Pro</div>'
                    '<div class="tag">Heart risk estimator</div>'
                    '<div style="margin-top:5px;font-size:11px;color:#aab8d0;">Developed by Shashant Sahu</div>'
                    '</div></div>',
                    unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"Signed in as **{user['full_name'] or user.get('email') or user['username']}**")
        if st.button("Log out", key="logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.caption("Educational project - not a medical diagnosis.")
    return page


# ----------------------------------------------------------------- Dashboard
def page_dashboard(user: dict, meta: dict, models: dict) -> None:
    st.title("Dashboard")
    st.caption(f"Welcome back, {user['full_name'] or user.get('email') or user['username']}.")
    rows = auth.list_assessments(user["id"])
    best = meta["best_model"]
    best_info = meta["models"][best]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Your assessments", str(len(rows)), "saved in your history"),
                unsafe_allow_html=True)
    if rows:
        latest = rows[0]
        c2.markdown(card("Latest estimate", f"{latest['probability'] * 100:.0f}%",
                         f"{latest['risk_level']} band - {fmt_dt(latest['created_at'])}", True),
                    unsafe_allow_html=True)
        avg = sum(a["probability"] for a in rows) / len(rows)
        c3.markdown(card("Average estimate", f"{avg * 100:.0f}%", "across all assessments"),
                    unsafe_allow_html=True)
    else:
        c2.markdown(card("Latest estimate", "-", "no assessment yet", True), unsafe_allow_html=True)
        c3.markdown(card("Average estimate", "-", "no assessment yet"), unsafe_allow_html=True)
    c4.markdown(card("Best model", best, f"validation ROC-AUC "
                                         f"{best_info['cv']['roc_auc']['mean']:.2f}"),
                unsafe_allow_html=True)

    if not rows:
        section("Get started")
        st.markdown('<div class="cs-note">You have not run an assessment yet. Enter a '
                    'patient\'s details to get a risk estimate, an explanation and a '
                    'downloadable report.</div>', unsafe_allow_html=True)
        st.button("Start your first assessment", type="primary", on_click=go_to,
                  args=("New Assessment",))
    else:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig_history(rows), use_container_width=True, key="dash_history")
        with right:
            st.plotly_chart(fig_levels(rows), use_container_width=True, key="dash_levels")
        section("Recent assessments")
        recent = pd.DataFrame([{
            "Date": fmt_dt(a["created_at"]), "Model": a["model_name"],
            "Risk": f"{a['probability'] * 100:.0f}%", "Band": a["risk_level"]}
            for a in rows[:5]])
        st.dataframe(recent, use_container_width=True, hide_index=True)
        st.button("New assessment", type="primary", on_click=go_to, args=("New Assessment",))

    section("Model snapshot")
    snap = st.columns(5)
    for col, metric in zip(snap, tm.METRICS):
        col.markdown(card(f"{METRIC_LABELS[metric]} (test)",
                          f"{best_info['test'][metric]:.2f}", best), unsafe_allow_html=True)
    disclaimer_box()


# ------------------------------------------------------------ New assessment
def feature_input(spec: dict):
    key = f"in_{spec['name']}"
    if spec["kind"] == "num":
        label = f"{spec['label']} ({spec['unit']})"
        if spec["dtype"] == "int":
            return st.number_input(label, min_value=int(spec["min"]), max_value=int(spec["max"]),
                                   value=int(spec["default"]), step=int(spec["step"]),
                                   help=spec["help"], key=key)
        return st.number_input(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                               value=float(spec["default"]), step=float(spec["step"]),
                               format="%.1f", help=spec["help"], key=key)
    options = list(spec["options"].keys())
    return st.selectbox(spec["label"], options, index=options.index(spec["default"]),
                        format_func=lambda k, o=spec["options"]: o[k], help=spec["help"], key=key)


def plain_language(result: dict, factors: pd.DataFrame) -> str:
    p, level = result["probability"], result["risk_level"]
    text = (f"For the details you entered, the **{result['model_name']}** model estimates a "
            f"**{p * 100:.0f}%** probability of heart disease, compared with patients in its "
            f"training data. That falls in the **{level.lower()}** band. {LEVEL_MEANING[level]}")
    up = [r.label for r in factors.itertuples() if r.influence > 1][:3]
    down = [r.label for r in factors.itertuples() if r.influence < -1][:3]
    if up:
        text += f" Values pushing the estimate **up** most: {', '.join(up)}."
    if down:
        text += f" Values pulling it **down** most: {', '.join(down)}."
    return text


def page_assessment(user: dict, meta: dict, models: dict) -> None:
    st.title("New risk assessment")
    st.caption("Enter the patient's details exactly as measured. All fields are required.")
    names = list(models)
    with st.form("assessment_form"):
        cols = st.columns(3)
        values = {}
        for i, spec in enumerate(tm.FEATURES):
            with cols[i % 3]:
                values[spec["name"]] = feature_input(spec)
        model_name = st.selectbox(
            "Prediction model", names, index=names.index(meta["best_model"]),
            format_func=lambda n: f"{n}  (best on validation)" if n == meta["best_model"] else n)
        submitted = st.form_submit_button("Calculate risk", type="primary")

    if submitted:
        try:
            clean = tm.validate_patient_input(values)
        except tm.InputValidationError as exc:
            for problem in exc.errors:
                st.error(problem)
        else:
            pipe = models[model_name]
            outcome = tm.predict_risk(pipe, clean)
            expl = tm.explain_prediction(pipe, meta, clean)
            factors = expl.head(8).to_dict("records")
            assessment_id = auth.save_assessment(
                user["id"], model_name, outcome["probability"], outcome["risk_level"],
                clean, factors)
            saved = auth.get_assessment(user["id"], assessment_id)
            st.session_state["last_result"] = {
                "user_id": user["id"], "assessment_id": assessment_id,
                "created_at": saved["created_at"], "model_name": model_name,
                "probability": outcome["probability"], "risk_level": outcome["risk_level"],
                "inputs": clean, "factors": factors,
                "all_probs": tm.predict_all_models(models, clean)}

    result = st.session_state.get("last_result")
    if result and result["user_id"] == user["id"]:
        render_result(user, meta, result)


def render_result(user: dict, meta: dict, result: dict) -> None:
    st.markdown("---")
    st.subheader("Result")
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(fig_gauge(result["probability"], result["risk_level"]),
                        use_container_width=True, key="res_gauge")
        st.markdown(f"<div style='text-align:center'>{pill(result['risk_level'])}</div>",
                    unsafe_allow_html=True)
    factors = pd.DataFrame(result["factors"])
    with right:
        st.markdown("#### What this means")
        st.markdown(plain_language(result, factors))
        agree = sum(1 for p in result["all_probs"].values() if p >= 0.5)
        st.markdown(f"**Model agreement:** {agree} of {len(result['all_probs'])} models "
                    f"estimate 50% or higher for this patient.")
        st.markdown('<div class="cs-note">Please discuss these values with a doctor, '
                    'especially if there are symptoms such as chest pain, breathlessness '
                    'or dizziness.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_factors(factors), use_container_width=True, key="res_factors")
    with c2:
        st.plotly_chart(fig_models(result["all_probs"], result["model_name"]),
                        use_container_width=True, key="res_models")
    st.caption("Red bars raise the estimate and green bars lower it, compared with a typical "
               "patient (median / most common category of the training data). This shows what "
               "the model relies on; it is not a medical cause.")

    table = factors.rename(columns={"label": "Factor", "value": "Your value",
                                    "typical": "Typical value", "influence": "Effect (pct. points)"})
    st.dataframe(table[["Factor", "Your value", "Typical value", "Effect (pct. points)"]],
                 use_container_width=True, hide_index=True)

    assessment = {k: result[k] for k in ("created_at", "model_name", "probability",
                                         "risk_level", "inputs", "factors")}
    assessment["id"] = result["assessment_id"]
    st.download_button("Download PDF report", data=reporting.build_pdf_report(user, assessment, meta),
                       file_name=f"cardioscore_report_{result['assessment_id']}.pdf",
                       mime="application/pdf", key=f"pdf_result_{result['assessment_id']}")
    st.caption(f"Saved to your history on {fmt_dt(result['created_at'])}.")
    disclaimer_box()


# ---------------------------------------------------------- History & reports
def page_history(user: dict, meta: dict, models: dict) -> None:
    st.title("History & Reports")
    rows = auth.list_assessments(user["id"])
    if not rows:
        st.info("No assessments yet. Run one from the 'New Assessment' page.")
        st.button("New assessment", type="primary", on_click=go_to, args=("New Assessment",))
        return

    table = pd.DataFrame([{
        "ID": a["id"], "Date": fmt_dt(a["created_at"]), "Model": a["model_name"],
        "Risk %": round(a["probability"] * 100, 1), "Band": a["risk_level"]} for a in rows])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Export all as CSV", data=reporting.assessments_to_csv(rows),
                       file_name="cardioscore_history.csv", mime="text/csv", key="csv_all")

    section("Assessment details")
    labels = {a["id"]: f"#{a['id']} - {fmt_dt(a['created_at'])} - {a['risk_level']} "
                       f"({a['probability'] * 100:.0f}%)" for a in rows}
    chosen = st.selectbox("Open an assessment", list(labels), format_func=lambda i: labels[i],
                          key="history_choice")
    a = auth.get_assessment(user["id"], chosen)  # always scoped to the logged-in user
    if a is None:
        st.warning("Assessment not found.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Risk estimate", f"{a['probability'] * 100:.0f}%", a["risk_level"], True),
                unsafe_allow_html=True)
    c2.markdown(card("Assessment date", fmt_dt(a["created_at"]).split(",")[0],
                     fmt_dt(a["created_at"]).split(",")[-1].strip()), unsafe_allow_html=True)
    c3.markdown(card("Model", a["model_name"], "used for this estimate"), unsafe_allow_html=True)
    info = meta["models"].get(a["model_name"])
    c4.markdown(card("Model test ROC-AUC", f"{info['test']['roc_auc']:.2f}" if info else "n/a",
                     "current trained version"), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Details entered**")
        st.dataframe(pd.DataFrame(tm.describe_inputs(a["inputs"]), columns=["Feature", "Value"]),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Top influencing factors**")
        if a["factors"]:
            fac = pd.DataFrame(a["factors"]).rename(columns={
                "label": "Factor", "value": "Value", "influence": "Effect (pct. points)"})
            st.dataframe(fac[["Factor", "Value", "Effect (pct. points)"]],
                         use_container_width=True, hide_index=True)
    st.download_button("Download PDF report", data=reporting.build_pdf_report(user, a, meta),
                       file_name=f"cardioscore_report_{a['id']}.pdf", mime="application/pdf",
                       key=f"pdf_hist_{a['id']}")
    with st.expander("Delete this assessment"):
        confirm = st.checkbox("I understand this cannot be undone", key=f"del_ok_{a['id']}")
        if st.button("Delete permanently", key=f"del_btn_{a['id']}", disabled=not confirm):
            auth.delete_assessment(user["id"], a["id"])
            if st.session_state.get("last_result", {}).get("assessment_id") == a["id"]:
                del st.session_state["last_result"]
            st.rerun()
    disclaimer_box()


# --------------------------------------------------------- Model performance
def comparison_table(meta: dict) -> pd.DataFrame:
    rows = []
    for name in meta["selection"]["ranking"]:
        m = meta["models"][name]
        rows.append({
            "Rank": m["cv_rank"],
            "Model": ("★ " if name == meta["best_model"] else "") + name,
            "Validation ROC-AUC": f"{m['cv']['roc_auc']['mean']:.3f} ± {m['cv']['roc_auc']['std']:.3f}",
            "Validation F1": f"{m['cv']['f1']['mean']:.3f}",
            "Train acc.": f"{m['train_accuracy']:.3f}",
            "Test acc.": f"{m['test']['accuracy']:.3f}",
            "Precision": f"{m['test']['precision']:.3f}",
            "Recall": f"{m['test']['recall']:.3f}",
            "F1": f"{m['test']['f1']:.3f}",
            "ROC-AUC": f"{m['test']['roc_auc']:.3f}"})
    return pd.DataFrame(rows)


def page_performance(user: dict, meta: dict, models: dict) -> None:
    st.title("Model Performance")
    cfg, best = meta["config"], meta["best_model"]
    st.markdown(f'<div class="cs-note"><b>How the best model is chosen:</b> '
                f'{esc(meta["selection"]["criterion"])} The test set '
                f'({cfg["n_test"]} patients) is used only for the final, honest report. '
                f'<b>Selected: {esc(best)}.</b></div>', unsafe_allow_html=True)

    section("All models side by side")
    st.dataframe(comparison_table(meta), use_container_width=True, hide_index=True)
    st.caption("Train acc. = accuracy on the data the model learned from. A large gap between "
               "train accuracy and validation/test results signals overfitting, which is why "
               "training accuracy is never used to pick the model.")

    left, right = st.columns(2)
    with left:
        metric = st.selectbox("Compare models on", tm.METRICS, index=4,
                              format_func=lambda m: METRIC_LABELS[m], key="perf_metric")
        st.plotly_chart(fig_metric_bars(meta, metric), use_container_width=True, key="perf_bars")
    with right:
        st.plotly_chart(fig_roc(meta), use_container_width=True, key="perf_roc")

    section("Results for each model")
    tabs = st.tabs(meta["selection"]["ranking"])
    for tab, name in zip(tabs, meta["selection"]["ranking"]):
        info = meta["models"][name]
        with tab:
            st.markdown(f"**{name}**" + ("  -  selected as best model" if name == best else ""))
            cols = st.columns(5)
            for col, metric in zip(cols, tm.METRICS):
                col.markdown(card(f"{METRIC_LABELS[metric]} (test)",
                                  f"{info['test'][metric]:.3f}",
                                  f"validation {info['cv'][metric]['mean']:.3f}"),
                             unsafe_allow_html=True)
            a, b = st.columns([1, 1])
            with a:
                st.plotly_chart(fig_confusion(info["confusion_matrix"]),
                                use_container_width=True, key=f"cm_{tm.slugify(name)}")
            with b:
                (tn, fp), (fn, tp) = info["confusion_matrix"]
                st.markdown(
                    f"**Reading the confusion matrix** ({tn + fp + fn + tp} test patients)\n\n"
                    f"- **{tp}** patients with disease were correctly flagged (true positives)\n"
                    f"- **{tn}** healthy patients were correctly cleared (true negatives)\n"
                    f"- **{fp}** healthy patients were wrongly flagged (false positives)\n"
                    f"- **{fn}** patients with disease were missed (false negatives)")
                gap = info["train_accuracy"] - info["cv"]["accuracy"]["mean"]
                if gap > 0.10:
                    st.warning(f"Training accuracy ({info['train_accuracy']:.2f}) is much higher "
                               f"than validation accuracy ({info['cv']['accuracy']['mean']:.2f}): "
                               "this model overfits.")
                else:
                    st.success(f"Training accuracy ({info['train_accuracy']:.2f}) and validation "
                               f"accuracy ({info['cv']['accuracy']['mean']:.2f}) are close.")
            cv_table = pd.DataFrame([{
                "Metric": METRIC_LABELS[m],
                f"Validation ({cfg['cv_folds']}-fold CV)": f"{info['cv'][m]['mean']:.3f} ± {info['cv'][m]['std']:.3f}",
                "Test set": f"{info['test'][m]:.3f}"} for m in tm.METRICS])
            st.dataframe(cv_table, use_container_width=True, hide_index=True)
    disclaimer_box()


# ------------------------------------------------------------ Explainability
def page_explainability(user: dict, meta: dict, models: dict) -> None:
    st.title("Explainability")
    st.markdown("Understand **what the model relies on** and **where it can go wrong**.")
    names = list(models)
    name = st.selectbox("Model", names, index=names.index(meta["best_model"]), key="xai_model")
    st.plotly_chart(fig_importance(meta["models"][name]["importance"]),
                    use_container_width=True, key="xai_importance")
    st.caption("Permutation importance: each feature is shuffled on the unseen test set and the "
               "drop in ROC-AUC is measured. Bigger drop = the model depends on that feature more. "
               "With a small test set the values are noisy; look at the ± bars.")
    section("How individual predictions are explained")
    st.markdown(
        "After every assessment, each of your inputs is replaced one at a time by a *typical* "
        "value (median or most common category of the training data) and the model is asked "
        "again. The change in probability is that input's influence. Positive values raise the "
        "estimate, negative values lower it. It is an approximation of the model's behaviour, "
        "not proof of a medical cause.")
    info = meta["models"][name]
    section(f"Metrics of {name}")
    cols = st.columns(5)
    for col, metric in zip(cols, tm.METRICS):
        col.markdown(card(METRIC_LABELS[metric], f"{info['test'][metric]:.2f}",
                          f"validation {info['cv'][metric]['mean']:.2f}"), unsafe_allow_html=True)
    section("Limitations")
    for item in tm.LIMITATIONS:
        st.markdown(f"- {item}")
    disclaimer_box()


# --------------------------------------------------------------------- About
def page_about(user: dict, meta: dict, models: dict) -> None:
    st.title("About CardioScore Pro")
    ds, cfg = meta["dataset"], meta["config"]
    st.markdown("CardioScore Pro estimates the probability of heart disease from clinical "
                "measurements, using models trained on the labelled **UCI Heart Disease "
                "(Cleveland)** dataset.")
    section("Dataset and training")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Patients", str(ds.get("rows_clean", "?")),
                     f"{ds.get('rows_raw', '?')} raw rows"), unsafe_allow_html=True)
    c2.markdown(card("Features", str(len(tm.FEATURE_NAMES)), "clinical inputs"),
                unsafe_allow_html=True)
    c3.markdown(card("Train / test", f"{cfg['n_train']} / {cfg['n_test']}", "stratified split"),
                unsafe_allow_html=True)
    rate = ds.get("disease_rate")
    c4.markdown(card("Disease rate", f"{rate * 100:.0f}%" if rate is not None else "?",
                     "of patients in dataset"), unsafe_allow_html=True)
    st.markdown(f"- Missing values handled by imputation inside the pipeline: "
                f"`{ds.get('missing_values') or 'none'}`\n"
                f"- Duplicates removed: {ds.get('duplicates_removed', 0)}\n"
                f"- Trained on: {fmt_dt(meta['trained_at'])} with scikit-learn "
                f"{meta['versions']['scikit_learn']}")
    section("Input features")
    st.dataframe(pd.DataFrame([{
        "Feature": s["label"], "Column": s["name"],
        "Allowed values": (f"{s['min']} - {s['max']} {s['unit']}" if s["kind"] == "num"
                           else ", ".join(f"{k} = {v}" for k, v in s["options"].items()))}
        for s in tm.FEATURES]), use_container_width=True, hide_index=True)
    section("Limitations")
    for item in tm.LIMITATIONS:
        st.markdown(f"- {item}")
    disclaimer_box()


PAGE_FUNCTIONS = {
    "Dashboard": page_dashboard,
    "New Assessment": page_assessment,
    "History & Reports": page_history,
    "Model Performance": page_performance,
    "Explainability": page_explainability,
    "About": page_about,
}


def main() -> None:
    load_css()
    meta, models = require_models()
    auth.init_db()
    if "user" not in st.session_state:
        render_auth()
        return
    user = st.session_state["user"]
    page = render_sidebar(user)
    PAGE_FUNCTIONS[page](user, meta, models)


if __name__ == "__main__":
    main()
