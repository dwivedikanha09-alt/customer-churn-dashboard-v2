import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(to right,#0f172a,#111827);
    color:white;
}

.big-title{
    font-size:55px;
    font-weight:800;
    text-align:center;
    color:white;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:30px;
    font-size:18px;
}

.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:20px;
    border:1px solid #334155;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
}

.stDataFrame{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown("<div class='big-title'>📈 Customer Churn Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Deploy Ready Machine Learning Dashboard using Streamlit</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deploy Ready Machine Learning Dashboard using Streamlit</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deploy Ready Machine Learning Dashboard using Streamlit</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("customer_churn_prediction_dataset.csv")

# ---------------------------------------------------
# AUTO COLUMN DETECTION
# ---------------------------------------------------
columns = df.columns.tolist()

target_col = None

for col in columns:
    if "churn" in col.lower():
        target_col = col
        break

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙ Dashboard Controls")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) > 0:
    selected_num = st.sidebar.selectbox(
        "Select Numeric Column",
        numeric_cols
    )

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Total Records", len(df))

with col2:
    st.metric("📊 Total Columns", len(df.columns))

with col3:
    st.metric("🧠 Missing Values", df.isnull().sum().sum())

st.markdown("---")

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------
if len(numeric_cols) > 0:

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x=selected_num,
            title=f"{selected_num} Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.box(
            df,
            y=selected_num,
            title=f"{selected_num} Box Plot",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------
if target_col is not None:

    st.subheader("🤖 Logistic Regression Model")

    ml_df = df.copy()

    le = LabelEncoder()

    for col in ml_df.columns:
        if ml_df[col].dtype == "object":
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))

    X = ml_df.drop(target_col, axis=1)
    y = ml_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    st.success(f"✅ Model Accuracy : {round(accuracy*100,2)}%")

# ---------------------------------------------------
# DATAFRAME
# ---------------------------------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:gray'>
    Designed with ❤️ using Streamlit
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)