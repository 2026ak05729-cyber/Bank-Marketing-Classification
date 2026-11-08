
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")
st.write(
    "Compare five machine learning classifiers "
    "for predicting term deposit subscriptions."
)

# --------------------------------------------------
# 2. Load saved models and preprocessing artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load(
            "models/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "models/decision_tree.pkl"
        ),
        "KNN": joblib.load(
            "models/knn.pkl"
        ),
        "Naive Bayes": joblib.load(
            "models/naive_bayes.pkl"
        ),
        "SVM": joblib.load(
            "models/svm.pkl"
        )
    }

    scaler = joblib.load("preprocessing/scaler.pkl")

    feature_columns = joblib.load(
        "preprocessing/feature_columns.pkl"
    )

    return models, scaler, feature_columns


models, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# 3. Dataset upload
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset dimensions:", df.shape)

    # Check required columns
    required_columns = feature_columns + ["y"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    # Prepare features and target
    X_test = df[feature_columns]
    y_test = df["y"].astype(int)

    # Apply the saved scaler
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # 4. Model selection
    # --------------------------------------------------

    st.header("Model Evaluation")

    selected_model = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )

    model = models[selected_model]

    if selected_model in [
        "Logistic Regression", "KNN", "SVM"
    ]:
        X_eval = X_test_scaled
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)

    # Get prediction scores for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_eval)[:, 1]
    else:
        y_score = model.decision_function(X_eval)

    # --------------------------------------------------
    # 5. Calculate evaluation metrics
    # --------------------------------------------------

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    st.subheader(f"Performance: {selected_model}")

    cols = st.columns(3)

    for i, (metric, value) in enumerate(metrics.items()):
        cols[i % 3].metric(metric, f"{value:.4f}")

    # --------------------------------------------------
    # 6. Confusion Matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------
    # 7. Classification Report
    # --------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4)
    )

    # --------------------------------------------------
    # 8. Compare all five models
    # --------------------------------------------------

    st.header("Comparison of All Five Models")

    comparison = []

    for name, clf in models.items():

        if name in [
            "Logistic Regression", "KNN", "SVM"
        ]:
            X_model = X_test_scaled
        else:
            X_model = X_test

        pred = clf.predict(X_model)

        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(X_model)[:, 1]
        else:
            score = clf.decision_function(X_model)

        comparison.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1": f1_score(
                y_test, pred, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, pred)
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df.set_index("Model").round(4),
        use_container_width=True
    )

else:
    st.info("Please upload a test CSV file to begin evaluation.")
