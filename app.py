import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ───────────────────────── page config ──────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
)

# ───────────────────────── custom CSS ───────────────────────────
st.markdown("""
<style>
/* --- overall theme tweaks --- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
}
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #e0e0e0 !important;
}
/* metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}
div[data-testid="stMetric"] label {
    color: #a0a0ff !important;
    font-weight: 600;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
}
/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
/* buttons */
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(102,126,234,0.45);
}
/* selectbox / radio */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the Excel dataset."""
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame):
    """
    Replicate the notebook pipeline:
      1. Feature engineering
      2. Encoding
      3. Train-test split & scaling
    Returns X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled,
            scaler, feature_names, target_mapping, df_display
    """
    df = df.copy()

    # ── Drop Duplicates & Macroeconomic Columns ──
    df.drop_duplicates(inplace=True)
    cols_to_drop = ['Unemployment rate', 'Inflation rate', 'GDP']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # ── 1. Feature Engineering ──
    df["Total_Approved"] = (
        df["Curricular units 1st sem (approved)"]
        + df["Curricular units 2nd sem (approved)"]
    )
    df["Avg_Grade"] = (
        df["Curricular units 1st sem (grade)"]
        + df["Curricular units 2nd sem (grade)"]
    ) / 2
    total_enrolled = (
        df["Curricular units 1st sem (enrolled)"]
        + df["Curricular units 2nd sem (enrolled)"]
    )
    df["Approval_Rate"] = np.where(
        total_enrolled > 0,
        df["Total_Approved"] / total_enrolled,
        0,
    )
    df["Total_Evaluations"] = (
        df["Curricular units 1st sem (evaluations)"]
        + df["Curricular units 2nd sem (evaluations)"]
    )

    # ── 2. Encoding ──
    target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    df["Target_Encoded"] = df["Target"].map(target_mapping)

    ohe_cols = [
        "Marital Status", "Application mode", "Course",
        "Daytime/evening attendance", "Previous qualification",
        "Nacionality", "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ]

    # Save unique values per OHE column (for prediction dropdowns)
    ohe_unique = {col: sorted(df[col].dropna().unique().tolist()) for col in ohe_cols}

    df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    le = LabelEncoder()
    binary_str_cols = [
        c for c in df_encoded.select_dtypes(include="object").columns
        if c != "Target"
    ]
    # Save per-column label encoder mappings (for prediction)
    le_mappings = {}
    for col in binary_str_cols:
        le.fit(df_encoded[col])
        le_mappings[col] = {cls: idx for idx, cls in enumerate(le.classes_)}
        df_encoded[col] = le.transform(df_encoded[col])

    df_encoded.drop(columns=["Target"], inplace=True)

    X = df_encoded.drop(columns=["Target_Encoded"])
    y = df_encoded["Target_Encoded"]

    # ── 3. Split & Scale ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler,
        list(X.columns),
        target_mapping,
        df,  # original df with engineered features (for EDA)
        ohe_cols,
        ohe_unique,
        binary_str_cols,
        le_mappings,
    )


@st.cache_resource(show_spinner=False)
def train_models(_X_train, _X_test, _y_train, _y_test,
                 _X_train_scaled, _X_test_scaled, feature_names):
    """Train all models and return results + fitted model dict."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="mlogloss", verbosity=0,
        )

    results = []
    fitted = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(_X_train_scaled, _y_train)
            y_pred = model.predict(_X_test_scaled)
        else:
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)

        acc = accuracy_score(_y_test, y_pred)
        prec = precision_score(_y_test, y_pred, average="weighted")
        rec = recall_score(_y_test, y_pred, average="weighted")
        f1 = f1_score(_y_test, y_pred, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })
        fitted[name] = {"model": model, "y_pred": y_pred}

    results_df = (
        pd.DataFrame(results)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, fitted


# ────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🎓 Student Dropout Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔬 EDA", "🤖 Model Training", "🔮 Predict"],
)

# ────────────────────────────────────────────────────────────────
#  LOAD DATA
# ────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.xlsx")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Please place **data.xlsx** in the project folder.")
    st.stop()

with st.spinner("Loading dataset …"):
    raw_df = load_data(DATA_PATH)

with st.spinner("Preparing features & models …"):
    (
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler, feature_names, target_mapping, display_df,
        _ohe_cols, _ohe_unique, _bin_cols, _le_mappings,
    ) = prepare_data(raw_df)

    results_df, fitted_models = train_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        feature_names,
    )

inv_target = {v: k for k, v in target_mapping.items()}


# ────────────────────────────────────────────────────────────────
#  HELPER : preprocess a single raw prediction row
# ────────────────────────────────────────────────────────────────

def preprocess_prediction_row(
    raw_row: dict,
    ohe_cols, ohe_unique, bin_cols, le_mappings,
    feature_names_list, scaler_obj, model_name,
):
    """
    Take a dict of raw (un-encoded) feature values,
    run the exact same preprocessing used during training,
    and return a DataFrame ready for model.predict().
    """
    row_df = pd.DataFrame([raw_row])

    # ── Feature Engineering ──
    row_df["Total_Approved"] = (
        row_df["Curricular units 1st sem (approved)"]
        + row_df["Curricular units 2nd sem (approved)"]
    )
    row_df["Avg_Grade"] = (
        row_df["Curricular units 1st sem (grade)"]
        + row_df["Curricular units 2nd sem (grade)"]
    ) / 2
    total_enrolled = (
        row_df["Curricular units 1st sem (enrolled)"]
        + row_df["Curricular units 2nd sem (enrolled)"]
    )
    row_df["Approval_Rate"] = np.where(
        total_enrolled > 0,
        row_df["Total_Approved"] / total_enrolled,
        0,
    )
    row_df["Total_Evaluations"] = (
        row_df["Curricular units 1st sem (evaluations)"]
        + row_df["Curricular units 2nd sem (evaluations)"]
    )

    # ── One-hot encoding ──
    row_df = pd.get_dummies(row_df, columns=ohe_cols, drop_first=True)

    # ── Label encoding for binary string columns ──
    for col in bin_cols:
        if col in row_df.columns:
            mapping = le_mappings[col]
            row_df[col] = row_df[col].map(mapping).fillna(0).astype(int)

    # ── Align columns with training features ──
    row_df = row_df.reindex(columns=feature_names_list, fill_value=0)

    # ── Scale if needed ──
    if model_name == "Logistic Regression":
        row_df = pd.DataFrame(
            scaler_obj.transform(row_df),
            columns=feature_names_list,
        )

    return row_df


# ────────────────────────────────────────────────────────────────
#  PAGE : Overview
# ────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Target classes", len(target_mapping))

    st.subheader("First 10 rows")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Data types & missing values")
    info_df = pd.DataFrame({
        "dtype": raw_df.dtypes.astype(str),
        "non-null": raw_df.notnull().sum(),
        "null": raw_df.isnull().sum(),
    })
    st.dataframe(info_df, use_container_width=True)

# ────────────────────────────────────────────────────────────────
#  PAGE : EDA
# ────────────────────────────────────────────────────────────────

elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")
    st.markdown("---")

    eda_choice = st.selectbox(
        "Select a chart",
        [
            "Target Distribution",
            "Age at Enrollment Distribution",
            "Admission Grade Distribution",
            "Top Previous Qualifications",
            "Course Distribution",
            "1st Semester Approved Units vs Target",
            "2nd Semester Approved Units vs Target",
            "Gender vs Target",
            "Numeric Feature Correlations",
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e2f")
    ax.set_facecolor("#1e1e2f")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if eda_choice == "Target Distribution":
        counts = raw_df["Target"].value_counts()
        bars = ax.bar(counts.index, counts.values, color=["#ff6b6b", "#51cf66", "#339af0"], edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, str(int(h)), ha="center", va="bottom", color="white", fontweight="bold")
        ax.set_title("Target Distribution", fontweight="bold")
        ax.set_xlabel("Target")
        ax.set_ylabel("Count")

    elif eda_choice == "Age at Enrollment Distribution":
        sns.histplot(raw_df["Age at enrollment"], bins=20, color="#74c0fc", ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_title("Age at Enrollment Distribution", fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")

    elif eda_choice == "Admission Grade Distribution":
        sns.histplot(raw_df["Admission grade"], bins=20, color="#ffa94d", ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_title("Admission Grade Distribution", fontweight="bold")
        ax.set_xlabel("Admission Grade")
        ax.set_ylabel("Count")

    elif eda_choice == "Top Previous Qualifications":
        counts = raw_df["Previous qualification"].value_counts().head(10)
        sns.barplot(x=counts.index, y=counts.values, palette="plasma", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Top 10 Previous Qualifications", fontweight="bold")
        ax.set_xlabel("Qualification")
        ax.set_ylabel("Count")

    elif eda_choice == "Course Distribution":
        course_counts = raw_df["Course"].value_counts()
        sns.barplot(x=course_counts.index, y=course_counts.values, palette="viridis", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_title("Course Distribution", fontweight="bold")
        ax.set_xlabel("Course")
        ax.set_ylabel("Count")

    elif eda_choice == "1st Semester Approved Units vs Target":
        sns.boxplot(x="Target", y="Curricular units 1st sem (approved)", data=raw_df, palette="Set2", ax=ax)
        ax.set_title("1st Semester Approved Units vs Target", fontweight="bold")

    elif eda_choice == "2nd Semester Approved Units vs Target":
        sns.boxplot(x="Target", y="Curricular units 2nd sem (approved)", data=raw_df, palette="Set2", ax=ax)
        ax.set_title("2nd Semester Approved Units vs Target", fontweight="bold")

    elif eda_choice == "Gender vs Target":
        sns.countplot(x="Gender", hue="Target", data=raw_df, palette="Set1", ax=ax)
        plt.setp(ax.get_legend().get_texts(), color='black')  # Fix legend text color
        ax.set_title("Gender vs Target", fontweight="bold")

    elif eda_choice == "Numeric Feature Correlations":
        numeric_df = raw_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig2, ax2 = plt.subplots(figsize=(14, 12))
        fig2.patch.set_facecolor("#1e1e2f")
        ax2.set_facecolor("#1e1e2f")
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax2, linewidths=0.3,
                    cbar_kws={"shrink": 0.6})
        ax2.set_title("Correlation Heatmap (Numeric Features)", color="white", fontweight="bold")
        ax2.tick_params(colors="white", labelsize=7)
        st.pyplot(fig2)
        fig = None  # skip generic pyplot below

    if fig is not None:
        fig.tight_layout()
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
#  PAGE : Model Training
# ────────────────────────────────────────────────────────────────

elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Comparison")
    st.markdown("---")

    st.subheader("Ranked Results")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        }).highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score"],
            color="#2b8a3e",
        ),
        use_container_width=True,
    )

    best = results_df.iloc[0]
    st.success(
        f"🏆 **Best model**: {best['Model']}  —  "
        f"Accuracy {best['Accuracy']:.4f} · F1 {best['F1-Score']:.4f}"
    )

    # Bar chart comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#1e1e2f")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#667eea", "#764ba2", "#f783ac", "#ffa94d"]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor("#1e1e2f")
        bars = ax.barh(results_df["Model"], results_df[metric], color=colors[i], edgecolor="white", linewidth=0.4)
        ax.set_title(metric, color="white", fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlim(0.7, max(results_df[metric]) + 0.02)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", color="white", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
#  PAGE : Results (Feature Importance & Confusion Matrices)
# ────────────────────────────────────────────────────────────────

elif page == "📈 Results":
    st.title("📈 Detailed Results")
    st.markdown("---")

    selected_model = st.selectbox("Select a model", list(fitted_models.keys()))

    model_obj = fitted_models[selected_model]["model"]
    y_pred = fitted_models[selected_model]["y_pred"]

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1e1e2f")
    ax.set_facecolor("#1e1e2f")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[inv_target[i] for i in sorted(inv_target)])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {selected_model}", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for t in ax.texts:
        t.set_color("white")
    fig.tight_layout()
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=[inv_target[i] for i in sorted(inv_target)],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)

    # Feature importance (tree-based)
    if hasattr(model_obj, "feature_importances_"):
        st.subheader("Top 20 Feature Importances")
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model_obj.feature_importances_,
        }).sort_values("Importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#1e1e2f")
        ax.set_facecolor("#1e1e2f")
        ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#ff8787", edgecolor="white", linewidth=0.4)
        ax.set_title(f"Top 20 Features — {selected_model}", color="white", fontweight="bold")
        ax.set_xlabel("Importance Score", color="white")
        ax.tick_params(colors="white", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importances are not available for this model type.")

# ────────────────────────────────────────────────────────────────
#  PAGE : Predict
# ────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Make a Prediction")
    st.markdown("---")
    st.info(
        "Fill in **all** features for a student below and click **Predict** to see "
        "the predicted outcome (Dropout / Enrolled / Graduate) using the best model."
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]["model"]

    # Helper to get unique sorted values from the raw data for dropdowns
    def _unique(col):
        return sorted(raw_df[col].dropna().unique().tolist())

    with st.form("prediction_form"):

        # ─── 1. Personal Information ───────────────────────────
        st.subheader("👤 Personal Information")
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input("Age at enrollment", 15, 70, 20)
            gender = st.selectbox("Gender", _unique("Gender"))
        with p2:
            marital_status = st.selectbox("Marital Status", _unique("Marital Status"))
            nacionality = st.selectbox("Nacionality", _unique("Nacionality"))
        with p3:
            international = st.selectbox("International", _unique("International"))
            displaced = st.selectbox("Displaced", _unique("Displaced"))

        st.markdown("---")

        # ─── 2. Academic Background ────────────────────────────
        st.subheader("📖 Academic Background")
        a1, a2, a3 = st.columns(3)
        with a1:
            application_mode = st.selectbox("Application mode", _unique("Application mode"))
            application_order = st.selectbox("Application order", _unique("Application order"))
        with a2:
            course = st.selectbox("Course", _unique("Course"))
            daytime = st.selectbox("Daytime/evening attendance", _unique("Daytime/evening attendance"))
        with a3:
            prev_qual = st.selectbox("Previous qualification", _unique("Previous qualification"))
            prev_qual_grade = st.number_input("Previous qualification (grade)", 0.0, 200.0, 130.0, step=0.1)
            admission_grade = st.number_input("Admission grade", 0.0, 200.0, 130.0, step=0.1)

        st.markdown("---")

        # ─── 3. Family Background ──────────────────────────────
        st.subheader("👨‍👩‍👦 Family Background")
        f1, f2 = st.columns(2)
        with f1:
            mother_qual = st.selectbox("Mother's qualification", _unique("Mother's qualification"))
            mother_occ = st.selectbox("Mother's occupation", _unique("Mother's occupation"))
        with f2:
            father_qual = st.selectbox("Father's qualification", _unique("Father's qualification"))
            father_occ = st.selectbox("Father's occupation", _unique("Father's occupation"))

        st.markdown("---")

        # ─── 4. Financial Information ──────────────────────────
        st.subheader("💰 Financial Information")
        fi1, fi2, fi3, fi4 = st.columns(4)
        with fi1:
            debtor = st.selectbox("Debtor", _unique("Debtor"))
        with fi2:
            tuition = st.selectbox("Tuition fees up to date", _unique("Tuition fees up to date"))
        with fi3:
            scholarship = st.selectbox("Scholarship holder", _unique("Scholarship holder"))
        with fi4:
            special_needs = st.selectbox("Educational special needs", _unique("Educational special needs"))

        st.markdown("---")

        # ─── 5. 1st Semester Performance ───────────────────────
        st.subheader("📝 1st Semester Performance")
        s1a, s1b, s1c = st.columns(3)
        with s1a:
            u1_credited = st.number_input("1st sem units credited", 0, 30, 0)
            u1_enrolled = st.number_input("1st sem units enrolled", 0, 30, 6)
        with s1b:
            u1_evaluations = st.number_input("1st sem evaluations", 0, 50, 6)
            u1_approved = st.number_input("1st sem units approved", 0, 30, 5)
        with s1c:
            u1_grade = st.number_input("1st sem grade", 0.0, 20.0, 12.0, step=0.1)
            u1_without = st.number_input("1st sem without evaluations", 0, 30, 0)

        st.markdown("---")

        # ─── 6. 2nd Semester Performance ───────────────────────
        st.subheader("📝 2nd Semester Performance")
        s2a, s2b, s2c = st.columns(3)
        with s2a:
            u2_credited = st.number_input("2nd sem units credited", 0, 30, 0)
            u2_enrolled = st.number_input("2nd sem units enrolled", 0, 30, 6)
        with s2b:
            u2_evaluations = st.number_input("2nd sem evaluations", 0, 50, 6)
            u2_approved = st.number_input("2nd sem units approved", 0, 30, 5)
        with s2c:
            u2_grade = st.number_input("2nd sem grade", 0.0, 20.0, 12.0, step=0.1)
            u2_without = st.number_input("2nd sem without evaluations", 0, 30, 0)

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        # Build a raw row dict matching original DataFrame columns
        raw_row = {
            "Marital Status": marital_status,
            "Application mode": application_mode,
            "Application order": application_order,
            "Course": course,
            "Daytime/evening attendance": daytime,
            "Previous qualification": prev_qual,
            "Previous qualification (grade)": prev_qual_grade,
            "Nacionality": nacionality,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
            "Admission grade": admission_grade,
            "Displaced": displaced,
            "Educational special needs": special_needs,
            "Debtor": debtor,
            "Tuition fees up to date": tuition,
            "Gender": gender,
            "Scholarship holder": scholarship,
            "Age at enrollment": age,
            "International": international,
            "Curricular units 1st sem (credited)": u1_credited,
            "Curricular units 1st sem (enrolled)": u1_enrolled,
            "Curricular units 1st sem (evaluations)": u1_evaluations,
            "Curricular units 1st sem (approved)": u1_approved,
            "Curricular units 1st sem (grade)": u1_grade,
            "Curricular units 1st sem (without evaluations)": u1_without,
            "Curricular units 2nd sem (credited)": u2_credited,
            "Curricular units 2nd sem (enrolled)": u2_enrolled,
            "Curricular units 2nd sem (evaluations)": u2_evaluations,
            "Curricular units 2nd sem (approved)": u2_approved,
            "Curricular units 2nd sem (grade)": u2_grade,
            "Curricular units 2nd sem (without evaluations)": u2_without,
        }

        processed = preprocess_prediction_row(
            raw_row,
            _ohe_cols, _ohe_unique, _bin_cols, _le_mappings,
            feature_names, scaler, best_name,
        )

        pred = best_model.predict(processed)[0]
        label = inv_target.get(int(pred), "Unknown")
        emoji = {"Dropout": "❌", "Enrolled": "📚", "Graduate": "🎓"}.get(label, "❓")

        st.markdown("---")
        st.markdown(
            f"<h2 style='text-align:center;'>{emoji} Predicted Outcome: "
            f"<span style='color:#51cf66;'>{label}</span></h2>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model used: **{best_name}**")
