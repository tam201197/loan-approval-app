import streamlit as st
import plotly.express as px
from utils import load_data, build_preprocessor, validate_models
from sklearn.metrics import confusion_matrix, roc_curve
import plotly.graph_objects as go

st.set_page_config(page_title="📊 Model Validation", layout="wide")
st.title("📊 Advanced Model Evaluation")

st.subheader("🖥 Training Console")

console_placeholder = st.empty()

logs = []

def logger(message):
    logs.append(message)
    console_placeholder.code("\n".join(logs))

df = load_data()
preprocessor = build_preprocessor(df)

if st.button("🚀 Run Full Validation"):

    results_df, best_model, best_name = validate_models(df, preprocessor, log_callback=logger)

    st.subheader("📊 Model Comparison")
    st.dataframe(results_df)

    # Highlight best
    best_row = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    st.success(f"🏆 Best Model: {best_name} (ROC-AUC: {best_row['ROC-AUC']:.3f})")

    # 🔥 Accuracy Chart
    fig = px.bar(
        results_df,
        x="Model",
        y="ROC-AUC",
        text="ROC-AUC"
    )
    st.plotly_chart(fig, width='stretch')

    # 🔥 Confusion Matrix of Best Model
    df_clean = df.dropna(subset=["loan_status"])
    X = df_clean.drop("loan_status", axis=1)
    y = df_clean["loan_status"]

    y_pred = best_model.predict(X)

    cm = confusion_matrix(y, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, width='stretch')

    # 🔥 ROC Curve
    y_prob = best_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random"))

    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig_roc, width='stretch')

    st.success("✅ Best model saved as best_model.pkl")