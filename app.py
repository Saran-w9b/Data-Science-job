# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Data Science Job Salaries — Demo", layout="wide")
sns.set_theme()

# ---------------------------
# Helpers / Caching
# ---------------------------
@st.cache_data
def load_data(path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path)
    return df

def clean_dataframe(df):
    df = df.copy()
    # drop unnamed index cols
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # drop duplicates
    df = df.drop_duplicates()
    # handle common missing values
    # if salary_in_usd present, keep it as salary; else keep salary
    if 'salary_in_usd' in df.columns and df['salary_in_usd'].notnull().any():
        df['salary'] = df['salary_in_usd']
    # Fill or drop NA for key columns
    df = df.dropna(subset=['salary', 'job_title', 'experience_level'])
    # Map experience levels
    mapping_exp = {'SE':'Senior','MI':'Mid','EN':'Entry','EX':'Executive',
                   'Senior':'Senior','Mid':'Mid','Entry':'Entry','Executive':'Executive'}
    df['experience_level'] = df['experience_level'].map(mapping_exp).fillna(df['experience_level'])
    # Map employment type
    mapping_emp = {'FT':'Full-time','PT':'Part-time','CT':'Contract','FL':'Freelance',
                   'Full-time':'Full-time','Part-time':'Part-time','Contract':'Contract','Freelance':'Freelance'}
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].map(mapping_emp).fillna(df['employment_type'])
    # Company size
    mapping_size = {'S':'Small','M':'Medium','L':'Large','Small':'Small','Medium':'Medium','Large':'Large'}
    if 'company_size' in df.columns:
        df['company_size'] = df['company_size'].map(mapping_size).fillna(df['company_size'])
    # Remote ratio -> job_type
    if 'remote_ratio' in df.columns and 'job_type' not in df.columns:
        df.rename(columns={'remote_ratio':'job_type'}, inplace=True)
    if 'job_type' in df.columns:
        df['job_type'] = df['job_type'].map({100:'remote', 0:'onsite', 50:'hybrid',
                                             'remote':'remote','onsite':'onsite','hybrid':'hybrid'}).fillna(df['job_type'])
    # ensure salary numeric
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df = df.dropna(subset=['salary'])
    df.reset_index(drop=True, inplace=True)
    return df

# ---------------------------
# App layout
# ---------------------------
st.title("📊 Data Science Job Salaries — Live Demo")
st.markdown("Upload the dataset or use a local file path. This demo shows EDA, trains models, and provides a salary predictor.")

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload CSV (ds_salaries or similar)", type=['csv'])
    local_path = st.text_input("Or enter local CSV path (ignored if upload used)", value="Data Science Job Salaries.csv")
    st.write("---")
    st.header("Modeling options")
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
    test_size = st.slider("Test set %", min_value=10, max_value=40, value=20)
    random_state = st.number_input("Random seed", value=42, step=1)
    st.write("---")
    st.markdown("Tip: If app is slow, upload a smaller CSV or sample the data below.")

# Load data
try:
    df_raw = load_data(path=local_path, uploaded_file=uploaded)
except Exception as e:
    st.error(f"Could not load file. Put CSV in path or upload. Error: {e}")
    st.stop()

df = clean_dataframe(df_raw)

# Quick overview
st.subheader("Data preview")
c1, c2 = st.columns([2,1])
with c1:
    st.dataframe(df.head(200))
with c2:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.write("Columns:", list(df.columns))

# EDA controls
st.subheader("Exploratory Data Analysis (select plot)")
eda_plot = st.selectbox("Plot", [
    "Salary distribution",
    "Salary by Experience level",
    "Salary by Employment type",
    "Salary by Company size",
    "Salary by Job type (remote/hybrid/onsite)",
    "Top roles by mean salary",
    "Top countries by job count"
])

# EDA plots
fig = plt.figure(figsize=(8,5))
if eda_plot == "Salary distribution":
    sns.histplot(df['salary'], bins=30, kde=True)
    plt.title("Salary distribution (USD)")
elif eda_plot == "Salary by Experience level":
    order = ['Entry','Mid','Senior','Executive']
    sns.boxplot(data=df, x='experience_level', y='salary', order=[o for o in order if o in df['experience_level'].unique()])
    plt.title("Salary by Experience level")
elif eda_plot == "Salary by Employment type":
    if 'employment_type' in df.columns:
        sns.boxplot(data=df, x='employment_type', y='salary')
        plt.title("Salary by Employment type")
    else:
        st.warning("employment_type column not in dataset.")
elif eda_plot == "Salary by Company size":
    if 'company_size' in df.columns:
        sns.boxplot(data=df, x='company_size', y='salary', order=['Small','Medium','Large'])
        plt.title("Salary by Company size")
    else:
        st.warning("company_size column not in dataset.")
elif eda_plot == "Salary by Job type (remote/hybrid/onsite)":
    if 'job_type' in df.columns:
        sns.boxplot(data=df, x='job_type', y='salary')
        plt.title("Salary by Job type")
    else:
        st.warning("job_type column not in dataset.")
elif eda_plot == "Top roles by mean salary":
    top_roles = df.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top_roles.values, y=top_roles.index)
    plt.xlabel("Mean salary")
    plt.title("Top roles by mean salary")
elif eda_plot == "Top countries by job count":
    if 'company_location' in df.columns:
        top_c = df['company_location'].value_counts().head(15)
        sns.barplot(x=top_c.values, y=top_c.index)
        plt.xlabel("Number of job postings")
        plt.title("Top countries by job count")
    else:
        st.warning("company_location column not in dataset.")

st.pyplot(fig)

# ---------------------------
# Modeling
# ---------------------------
st.markdown("---")
st.subheader("Model training & evaluation")

# Choose features
# We will use a pragmatic subset of columns often present:
candidate_features = []
for col in ['experience_level', 'employment_type', 'job_title', 'job_type', 'company_size', 'company_location', 'work_year']:
    if col in df.columns:
        candidate_features.append(col)

st.write("Detected candidate features:", candidate_features)
features_selected = st.multiselect("Select features to use for model", candidate_features, default=candidate_features)

if len(features_selected) == 0:
    st.warning("Select at least one feature.")
    st.stop()

# Prepare X, y
X = df[features_selected].copy()
y = df['salary'].copy()

# basic preprocessing: fill NaNs in categorical with 'Unknown'
for c in X.select_dtypes(include=['object']).columns:
    X[c] = X[c].fillna('Unknown')

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))

# Build pipeline: OneHot for categorical, passthrough for numeric
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

transformers = []
if categorical_cols:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

if model_choice == "Linear Regression":
    model = Pipeline(steps=[('pre', preprocessor), ('reg', LinearRegression())])
else:
    model = Pipeline(steps=[('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=100, random_state=int(random_state)))])

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success("Training complete")
st.write("Evaluation on test set:")
st.metric("MAE", f"{mae:,.2f}")
st.metric("MSE", f"{mse:,.2f}")
st.metric("R²", f"{r2:.3f}")

st.write("Sample predictions (first 10):")
pred_df = pd.DataFrame({
    "actual": y_test.values[:10],
    "predicted": y_pred[:10]
})
st.dataframe(pred_df)

# If Random Forest, show simple feature importances (requires encoding)
if model_choice == "Random Forest":
    try:
        # Extract feature names from preprocessor and show importances
        pre = model.named_steps['pre']
        ohe = pre.named_transformers_.get('cat', None)
        feature_names = []
        if ohe is not None:
            cat_cols = categorical_cols
            ohe_names = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(ohe_names)
        feature_names.extend(numeric_cols)
        importances = model.named_steps['reg'].feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
        st.write("Top feature importances:")
        st.dataframe(fi)
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

# ---------------------------
# Live predictor UI
# ---------------------------
st.markdown("---")
st.subheader("Interactive salary predictor (use the trained model)")

def single_row_from_inputs(features_selected):
    row = {}
    for f in features_selected:
        if f == 'experience_level':
            opts = sorted(df['experience_level'].unique().tolist())
            row[f] = st.selectbox("Experience level", opts, key='pred_exp')
        elif f == 'employment_type':
            opts = sorted(df['employment_type'].unique().tolist() if 'employment_type' in df.columns else ['Full-time','Part-time'])
            row[f] = st.selectbox("Employment type", opts, key='pred_emp')
        elif f == 'job_title':
            # allow typing or select top roles
            roles = df['job_title'].value_counts().index.tolist()
            row[f] = st.selectbox("Job title", options=roles[:200], index=0, key='pred_title')
        elif f == 'job_type':
            opts = sorted(df['job_type'].unique().tolist() if 'job_type' in df.columns else ['remote','onsite','hybrid'])
            row[f] = st.selectbox("Job type (remote/hybrid/onsite)", opts, key='pred_jtype')
        elif f == 'company_size':
            opts = sorted(df['company_size'].unique().tolist() if 'company_size' in df.columns else ['Small','Medium','Large'])
            row[f] = st.selectbox("Company size", opts, key='pred_csize')
        elif f == 'company_location':
            opts = sorted(df['company_location'].value_counts().index.tolist())
            row[f] = st.selectbox("Company location (country)", opts[:200], key='pred_cloc')
        elif f == 'work_year':
            vals = sorted(df['work_year'].unique().tolist())
            row[f] = st.selectbox("Work year", vals, index=len(vals)-1, key='pred_year')
        else:
            # fallback generic text
            row[f] = st.text_input(f, key=f'pred_{f}')
    return pd.DataFrame([row])

pred_input = single_row_from_inputs(features_selected)
st.write("Inputs:")
st.dataframe(pred_input)

if st.button("Predict salary"):
    try:
        # prepare same encoding as training
        X_template = pd.concat([X_train.head(0), pred_input], ignore_index=True, sort=False)
        X_template = X_template.fillna('Unknown')
        # One-hot via the pipeline preprocessor by calling transform
        pred_val = model.predict(pred_input)[0]
        st.success(f"Predicted salary (USD): ${pred_val:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write("---")
st.markdown("**Notes / tips for the demo:**")
st.markdown("""
- Put dataset in same folder or upload it. The app detects common columns described in the dataset.
- For better predictions, pick features that matter (experience_level, job_title, company_size, job_type).
- Random Forest tends to give better R² at the cost of interpretability.
""")

