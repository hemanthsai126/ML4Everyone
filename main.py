import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder


# ------------------------------
# Helper Functions
# ------------------------------

def clean_data(df, missing_strategy):
    """
    Remove duplicates and handle missing values.
    - If "Drop" is selected, rows with missing values are dropped.
    - Otherwise, numeric columns are filled with the mean and non-numeric with the mode.
    """
    df = df.drop_duplicates()
    if missing_strategy == "Drop":
        df = df.dropna()
    else:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def encode_features(df, target, problem_type):
    """
    Separates features and target.
    - For features, performs one-hot encoding.
    - For classification targets, forces conversion to string then applies label encoding
      so that the classes are mapped to contiguous integers.
    """
    X = df.drop(columns=[target])
    y = df[target]

    if problem_type == "classification":
        le = LabelEncoder()
        # Force conversion to string to avoid issues when numeric values are stored as objects.
        y = le.fit_transform(y.astype(str))
    return pd.get_dummies(X, drop_first=True), y


# ------------------------------
# Streamlit App
# ------------------------------

st.title("ML for Everyone")
st.write("Upload your data file (CSV or Excel) to begin.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("Error reading file: " + str(e))
    else:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # ------------------------------
        # Data Cleaning Section
        # ------------------------------
        st.subheader("Data Cleaning Options")
        missing_strategy = st.radio(
            "How do you want to handle missing values?",
            options=["Drop", "Fill with Mean"],
            index=0,
            help="Drop rows with missing values or Fill numeric columns with mean and categorical with mode."
        )
        st.info("Duplicates will be removed automatically.")
        df_clean = clean_data(df, missing_strategy)
        st.write("Cleaned Data Preview:")
        st.dataframe(df_clean.head())

        # ------------------------------
        # Target Selection and Problem Type Detection
        # ------------------------------
        st.subheader("Target Selection")
        target = st.selectbox("Select the column you want to predict", df_clean.columns)

        # Determine problem type using a simple heuristic:
        # If the target is numeric and has fewer than 10 unique values, treat it as classification.
        # Otherwise, if non-numeric, treat it as classification.
        if pd.api.types.is_numeric_dtype(df_clean[target]):
            if df_clean[target].nunique() < 10:
                problem_type = "classification"
            else:
                problem_type = "regression"
        else:
            problem_type = "classification"
        st.write("**Detected problem type:**", problem_type)

        # ------------------------------
        # Model Evaluation Section
        # ------------------------------
        st.subheader("Model Evaluation Options")
        evaluation_method = st.radio("Choose evaluation method", options=["Train/Test Split", "Cross Validation"])
        if evaluation_method == "Train/Test Split":
            test_size = st.slider("Test size fraction", 0.1, 0.5, 0.2)
        else:
            cv_folds = st.number_input("Number of folds for cross-validation", min_value=2, max_value=10, value=5,
                                       step=1)

        # Encode features and target accordingly.
        X, y = encode_features(df_clean, target, problem_type)

        # Split the data if using Train/Test Split.
        if evaluation_method == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, y_train = X, y  # Use full data for cross-validation

        # Define models based on problem type.
        models = {}
        if problem_type == "classification":
            models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
            models["Random Forest"] = RandomForestClassifier(random_state=42)
            models["AdaBoost"] = AdaBoostClassifier(random_state=42)
        else:
            models["Decision Tree"] = DecisionTreeRegressor(random_state=42)
            models["Random Forest"] = RandomForestRegressor(random_state=42)
            models["AdaBoost"] = AdaBoostRegressor(random_state=42)

        st.write("### Model Performance Scores")
        results = {}
        for name, model in models.items():
            if evaluation_method == "Train/Test Split":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if problem_type == "classification":
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
            else:
                if problem_type == "classification":
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                score = scores.mean()
            results[name] = score

        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Score"])
        st.dataframe(results_df.sort_values(by="Score", ascending=False))

        # ------------------------------
        # Prediction on Unseen Data Section
        # ------------------------------
        st.subheader("Make Predictions on Unseen Data")
        if st.checkbox("I want to predict on new (unseen) data"):
            chosen_model_name = st.selectbox("Select the model to use for prediction", list(models.keys()))
            # Retrain the chosen model on the entire dataset.
            chosen_model = models[chosen_model_name]
            chosen_model.fit(X, y)

            prediction_method = st.radio("Select prediction input method", options=["Single Row Input", "Upload CSV"])
            if prediction_method == "Single Row Input":
                st.write("Enter values for the features:")
                # Get original feature names (before encoding)
                feature_cols = list(df_clean.drop(columns=[target]).columns)
                input_data = {}
                for col in feature_cols:
                    input_val = st.text_input(f"Enter value for **{col}**")
                    input_data[col] = input_val

                if st.button("Predict for Single Row"):
                    input_df = pd.DataFrame([input_data])
                    # Convert columns to numeric if needed.
                    for col in input_df.columns:
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            try:
                                input_df[col] = pd.to_numeric(input_df[col])
                            except Exception as e:
                                st.error(f"Invalid numeric value for {col}: {e}")
                    input_encoded = pd.get_dummies(input_df, drop_first=True)
                    # Align input data columns with training data.
                    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
                    prediction = chosen_model.predict(input_encoded)
                    st.success("Prediction: " + str(prediction[0]))

            else:
                uploaded_new = st.file_uploader("Upload CSV or Excel file for prediction", type=["csv", "xlsx"],
                                                key="newdata")
                if uploaded_new is not None:
                    try:
                        if uploaded_new.name.endswith('.csv'):
                            new_df = pd.read_csv(uploaded_new)
                        else:
                            new_df = pd.read_excel(uploaded_new)
                    except Exception as e:
                        st.error("Error reading the new file: " + str(e))
                    else:
                        expected_features = set(df_clean.drop(columns=[target]).columns)
                        new_features = set(new_df.columns)
                        if expected_features != new_features:
                            st.error("Uploaded data columns do not match the original features. Expected columns: " +
                                     ", ".join(expected_features))
                        else:
                            new_encoded = pd.get_dummies(new_df, drop_first=True)
                            new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)
                            predictions = chosen_model.predict(new_encoded)
                            new_df["Prediction"] = predictions
                            st.write("Predictions:")
                            st.dataframe(new_df)

                            # Provide download options.
                            csv_data = new_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download predictions as CSV",
                                data=csv_data,
                                file_name='predictions.csv',
                                mime='text/csv'
                            )
                            towrite = io.BytesIO()
                            new_df.to_excel(towrite, index=False, engine='openpyxl')
                            towrite.seek(0)
                            st.download_button(
                                label="Download predictions as Excel",
                                data=towrite,
                                file_name='predictions.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
