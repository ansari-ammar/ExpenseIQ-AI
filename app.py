
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="ExpenseIQ AI",
    page_icon="💰",
    layout="wide"
)

# Load model and datasets
model = joblib.load("models/expense_model.pkl")

df = pd.read_csv("cleaned_dataset.csv")
monthly = pd.read_csv("monthly_dataset.csv")
results = pd.read_csv("model_results.csv")
predictions = pd.read_csv("predictions.csv")

# Website styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
[data-testid="stMetric"] {
    background-color: #1b2330;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #303846;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("💰 ExpenseIQ AI")

st.sidebar.caption(
    "AI-Based Personal Expense Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Expense Prediction",
        "Data Analytics",
        "Model Performance",
        "Dataset",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.write("Machine Learning Project")
st.sidebar.write("BCA - IBM AI")

# HOME PAGE
if page == "Home":

    st.title("💰 ExpenseIQ AI")

    st.subheader(
        "Intelligent Personal Expense Prediction"
    )

    st.info(
        "Explore historical spending and generate "
        "experimental machine learning predictions."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", len(df))
    c2.metric("Historical Months", len(monthly))

    c3.metric(
        "Average Monthly Income",
        f"{monthly['monthly_income'].mean():,.2f}"
    )

    c4.metric(
        "Average Monthly Expense",
        f"{monthly['total_expense'].mean():,.2f}"
    )

    st.subheader("Monthly Expense Trends")

    fig = px.line(
        monthly,
        x="Month_Year",
        y="total_expense",
        markers=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Income vs Expenses")

    fig = px.line(
        monthly,
        x="Month_Year",
        y=["monthly_income", "total_expense"],
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)


# EXPENSE PREDICTION PAGE
elif page == "Expense Prediction":

    st.title("🔮 AI Expense Prediction")

    st.write(
        "Enter your expected monthly income "
        "to generate an experimental prediction."
    )

    income = st.number_input(
        "Expected Monthly Income",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    next_month = int(
        monthly["month_index"].max() + 1
    )

    if st.button("Predict Monthly Expense"):

        input_data = pd.DataFrame({
            "month_index": [next_month],
            "monthly_income": [income]
        })

        expense = max(
            0, float(model.predict(input_data)[0])
        )

        savings = income - expense

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Monthly Income",
            f"{income:,.2f}"
        )

        c2.metric(
            "Predicted Expense",
            f"{expense:,.2f}"
        )

        c3.metric(
            "Estimated Savings",
            f"{savings:,.2f}"
        )

        if income > 0:

            ratio = expense / income * 100

            st.subheader("Budget Status")

            st.progress(
                min(max(ratio / 100, 0), 1)
            )

            st.write(
                f"Expense to Income Ratio: {ratio:.2f}%"
            )

            if ratio <= 50:
                st.success("Low Spending Ratio")

            elif ratio <= 70:
                st.warning("Moderate Spending Ratio")

            else:
                st.error("High Spending Ratio")

            if savings >= 0:

                chart_data = pd.DataFrame({
                    "Category": ["Expenses", "Savings"],
                    "Amount": [expense, savings]
                })

                fig = px.pie(
                    chart_data,
                    names="Category",
                    values="Amount",
                    hole=0.5,
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig, use_container_width=True
                )

            else:
                st.warning(
                    "Predicted expenses exceed income."
                )

        else:
            st.warning(
                "Income must be positive to calculate a ratio."
            )

        st.caption(
            "Budget status uses predefined rules, "
            "not a trained classification model."
        )

        st.warning(
            "Experimental prediction only. "
            "The model has limited predictive accuracy."
        )


# DATA ANALYTICS PAGE
elif page == "Data Analytics":

    st.title("📊 Data Analytics")

    expenses = df[df["Type"] == "Expense"]

    category = expenses.groupby(
        "Category"
    )["Amount"].sum().reset_index()

    st.subheader("Category-wise Spending")

    fig = px.pie(
        category,
        names="Category",
        values="Amount",
        hole=0.4,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Expense Category Comparison")

    fig = px.bar(
        category,
        x="Category",
        y="Amount",
        color="Category",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    st.image(
        "charts/correlation_heatmap.png"
    )


# MODEL PERFORMANCE PAGE
elif page == "Model Performance":

    st.title("🧠 Model Performance")

    st.subheader("Selected Model")

    st.write(results["Model"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE", round(results["MAE"].iloc[0], 2)
    )

    c2.metric(
        "MSE", round(results["MSE"].iloc[0], 2)
    )

    c3.metric(
        "RMSE", round(results["RMSE"].iloc[0], 2)
    )

    c4.metric(
        "R² Score", round(results["R2"].iloc[0], 4)
    )

    if results["R2"].iloc[0] < 0:

        st.warning(
            "Negative R² indicates that the model "
            "performed worse than predicting the "
            "mean of the test target values."
        )

    st.subheader("Actual vs Predicted Expenses")

    fig = px.line(
        predictions,
        x="Month",
        y=["Actual", "Predicted"],
        markers=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        predictions,
        use_container_width=True
    )


# DATASET PAGE
elif page == "Dataset":

    st.title("📁 Dataset Explorer")

    st.subheader("Cleaned Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False),
        "cleaned_dataset.csv",
        "text/csv"
    )

    st.subheader("Monthly Dataset")

    st.dataframe(
        monthly,
        use_container_width=True
    )

    st.download_button(
        "Download Monthly Dataset",
        monthly.to_csv(index=False),
        "monthly_dataset.csv",
        "text/csv"
    )


# ABOUT PAGE
elif page == "About Project":

    st.title("About ExpenseIQ AI")

    st.write("""
    ExpenseIQ AI is a machine learning-based
    personal expense prediction system.

    The project analyses historical financial
    transactions and generates experimental
    monthly expense predictions.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Seaborn
    - Plotly
    - Streamlit
    - Joblib
    """)

    st.subheader("Project Methodology")

    st.write("""
    1. Dataset Collection
    2. Data Cleaning
    3. Categorical Encoding
    4. Monthly Aggregation
    5. Data Visualization
    6. Correlation Analysis
    7. Model Selection
    8. Model Training
    9. Model Evaluation
    10. Dashboard Development
    """)

    st.subheader("Developed By")

    st.write("Ammarus Salam Ansari")
    st.write("Enrollment No: UU252010175")
    st.write("BCA - IBM Artificial Intelligence")
    st.write("United University, Prayagraj")

st.divider()
st.caption("ExpenseIQ AI | Academic Machine Learning Project")

if page == "Model Performance":

    st.divider()

    st.subheader("Advanced Model Comparison")

    comparison = pd.read_csv("model_comparison.csv")

    st.dataframe(comparison, use_container_width=True)

    fig = px.bar(
        comparison,
        x="Model",
        y="Validation_MAE",
        color="Model",
        title="Machine Learning Model Comparison",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    improved = pd.read_csv("improved_results.csv")

    st.subheader("Advanced Experiment Results")

    st.dataframe(improved, use_container_width=True)

    st.info(
        "Historical Mean achieved the lowest "
        "validation error in the advanced experiment."
    )

    st.warning(
        "The dataset has limited predictive patterns. "
        "Predictions are for academic demonstration."
    )