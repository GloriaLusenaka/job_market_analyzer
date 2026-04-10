import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Page config
st.set_page_config(page_title="Salary Scout", layout="wide")

# Load data


@st.cache_data
def load_data():
    df = pd.read_csv("DataScience_salaries_2024.csv")
    return df


df = load_data()

# Sidebar navigation
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Go to", [
                        "📊 Role Comparison", "🌍 Location Analysis", "🤖 Salary Predictor", "📈 Insights Dashboard"])

# Helper function for filtering


def apply_filters(df, year=None, exp_level=None, remote=None):
    filtered = df.copy()
    if year and year != "All":
        filtered = filtered[filtered['work_year'] == year]
    if exp_level and exp_level != "All":
        filtered = filtered[filtered['experience_level'] == exp_level]
    if remote and remote != "All":
        filtered = filtered[filtered['remote_ratio'] == remote]
    return filtered


# ========== PAGE 1: ROLE COMPARISON ==========
if page == "📊 Role Comparison":
    st.title("Data Science Role Comparison")
    st.markdown("*Compare roles, analyze locations, and predict your salary*")

    col1, col2, col3 = st.columns(3)
    with col1:
        year_filter = st.selectbox(
            "Year", ["All"] + sorted(df['work_year'].unique()))
    with col2:
        exp_filter = st.selectbox(
            "Experience Level", ["All"] + sorted(df['experience_level'].unique()))
    with col3:
        size_filter = st.selectbox(
            "Company Size", ["All"] + sorted(df['company_size'].unique()))

    filtered_df = apply_filters(df, year_filter, exp_filter, None)
    if size_filter != "All":
        filtered_df = filtered_df[filtered_df['company_size'] == size_filter]

    top_roles = filtered_df['job_title'].value_counts().head(10).index
    role_df = filtered_df[filtered_df['job_title'].isin(top_roles)]

    st.subheader("💰 Salary Distribution by Role")
    fig = px.box(role_df, x='job_title', y='salary_in_usd', color='experience_level',
                 title="Salary Distribution by Job Title",
                 labels={'salary_in_usd': 'Salary (USD)', 'job_title': 'Job Title'})
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Average Salary by Role and Experience Level")
    avg_salary = role_df.groupby(['job_title', 'experience_level'])[
        'salary_in_usd'].mean().reset_index()
    fig2 = px.bar(avg_salary, x='job_title', y='salary_in_usd', color='experience_level',
                  barmode='group', title="Average Salary by Role & Experience",
                  labels={'salary_in_usd': 'Avg Salary (USD)', 'job_title': 'Job Title'})
    fig2.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Salary Summary Table")
    summary = role_df.groupby('job_title').agg(
        Count=('salary_in_usd', 'count'),
        Mean_Salary=('salary_in_usd', 'mean'),
        Median_Salary=('salary_in_usd', 'median'),
        Min_Salary=('salary_in_usd', 'min'),
        Max_Salary=('salary_in_usd', 'max')
    ).round(0).sort_values('Mean_Salary', ascending=False)
    st.dataframe(summary, use_container_width=True)

# ========== PAGE 2: LOCATION ANALYSIS ==========
elif page == "🌍 Location Analysis":
    st.title("🌍 Global Salary Analysis by Location")
    st.markdown("Explore how salaries vary by country and remote work status.")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of countries to show", 5, 30, 15)
    with col2:
        min_samples = st.slider("Minimum samples per country", 5, 50, 10)

    country_stats = df.groupby('company_location').agg(
        count=('salary_in_usd', 'count'),
        mean_salary=('salary_in_usd', 'mean'),
        median_salary=('salary_in_usd', 'median')
    ).reset_index()

    country_stats = country_stats[country_stats['count'] >= min_samples]
    top_countries = country_stats.nlargest(top_n, 'mean_salary')

    st.subheader("💵 Top Paying Countries")
    fig1 = px.bar(top_countries, x='company_location', y='mean_salary',
                  color='mean_salary', color_continuous_scale='Viridis',
                  title=f"Top {top_n} Countries by Average Salary",
                  labels={'company_location': 'Country', 'mean_salary': 'Avg Salary (USD)'})
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🗺️ Global Salary Heatmap")
    fig2 = px.choropleth(country_stats,
                         locations='company_location',
                         locationmode='country names',
                         color='mean_salary',
                         hover_name='company_location',
                         title="Average Data Science Salary by Country",
                         color_continuous_scale='Viridis')
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏠 Remote Work Impact on Salary")
    remote_impact = df.groupby(['company_location', 'remote_ratio'])[
        'salary_in_usd'].mean().reset_index()
    remote_impact = remote_impact[remote_impact['company_location'].isin(
        top_countries['company_location'].head(10))]

    fig3 = px.bar(remote_impact, x='company_location', y='salary_in_usd', color='remote_ratio',
                  barmode='group', title="Salary by Remote Ratio (Top 10 Countries)",
                  labels={'company_location': 'Country', 'salary_in_usd': 'Avg Salary (USD)', 'remote_ratio': 'Remote %'})
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# ========== PAGE 3: SALARY PREDICTOR ==========
elif page == "🤖 Salary Predictor":
    st.title("🤖 AI Salary Predictor")
    st.markdown(
        "Predict your expected salary based on job title, experience, location, and remote work.")

    @st.cache_resource
    def train_model():
        model_df = df.dropna(subset=['salary_in_usd'])

        le_job = LabelEncoder()
        le_exp = LabelEncoder()
        le_loc = LabelEncoder()
        le_size = LabelEncoder()
        le_remote = LabelEncoder()

        model_df['job_encoded'] = le_job.fit_transform(model_df['job_title'])
        model_df['exp_encoded'] = le_exp.fit_transform(
            model_df['experience_level'])
        model_df['loc_encoded'] = le_loc.fit_transform(
            model_df['company_location'])
        model_df['size_encoded'] = le_size.fit_transform(
            model_df['company_size'])
        model_df['remote_encoded'] = le_remote.fit_transform(
            model_df['remote_ratio'].astype(str))

        features = ['job_encoded', 'exp_encoded', 'loc_encoded',
                    'size_encoded', 'remote_encoded', 'work_year']
        X = model_df[features]
        y = model_df['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, le_job, le_exp, le_loc, le_size, le_remote, mae, r2

    model, le_job, le_exp, le_loc, le_size, le_remote, mae, r2 = train_model()

    st.info(f"📊 Model Performance: MAE = ${mae:,.0f} | R² Score = {r2:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Job Title", sorted(df['job_title'].unique()))
        experience = st.selectbox("Experience Level",
                                  [("Entry Level", "EN"), ("Mid Level", "MI"),
                                   ("Senior Level", "SE"), ("Executive", "EX")],
                                  format_func=lambda x: x[0])
        company_location = st.selectbox(
            "Company Location", sorted(df['company_location'].unique()))

    with col2:
        company_size = st.selectbox("Company Size",
                                    [("Small (<50 employees)", "S"),
                                     ("Medium (50-250)", "M"), ("Large (>250)", "L")],
                                    format_func=lambda x: x[0])
        remote = st.selectbox("Remote Work",
                              [("0% (On-site)", 0), ("50% (Hybrid)", 50), ("100% (Remote)", 100)])
        year = st.slider("Year", 2020, 2024, 2024)

    if st.button("🔮 Predict My Salary", type="primary"):
        job_enc = le_job.transform([job_title])[0]
        exp_enc = le_exp.transform([experience[1]])[0]
        loc_enc = le_loc.transform([company_location])[0]
        size_enc = le_size.transform([company_size[1]])[0]
        remote_enc = le_remote.transform([str(remote)])[0]

        features = np.array(
            [[job_enc, exp_enc, loc_enc, size_enc, remote_enc, year]])
        prediction = model.predict(features)[0]

        st.success(f"### 💰 Estimated Annual Salary: ${prediction:,.0f}")
        st.caption(
            f"Range estimate: ${prediction - mae:,.0f} - ${prediction + mae:,.0f}")

        st.subheader("📊 Similar Roles Comparison")
        similar = df[
            (df['experience_level'] == experience[1]) &
            (df['company_size'] == company_size[1])
        ].groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(8, 4))
        similar.plot(kind='barh', ax=ax, color='skyblue')
        ax.axvline(prediction, color='red', linestyle='--',
                   label=f'Your Prediction: ${prediction:,.0f}')
        ax.set_xlabel('Average Salary (USD)')
        ax.set_title('Average Salaries for Similar Roles')
        ax.legend()
        st.pyplot(fig)

# ========== PAGE 4: INSIGHTS DASHBOARD ==========
else:
    st.title("📈 Key Insights Dashboard")
    st.markdown("Automatically generated insights from the data.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Salary", f"${df['salary_in_usd'].mean():,.0f}")
    with col3:
        st.metric("Median Salary", f"${df['salary_in_usd'].median():,.0f}")
    with col4:
        st.metric("Unique Job Titles", df['job_title'].nunique())

    st.subheader("📈 Salary Trend Over Time")
    yearly_trend = df.groupby('work_year')['salary_in_usd'].agg(
        ['mean', 'median']).reset_index()
    fig1 = px.line(yearly_trend, x='work_year', y=['mean', 'median'],
                   title="Average vs Median Salary by Year",
                   labels={'value': 'Salary (USD)', 'work_year': 'Year', 'variable': 'Metric'})
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Experience Level Distribution")
        exp_counts = df['experience_level'].value_counts()
        fig2 = px.pie(values=exp_counts.values,
                      names=exp_counts.index, title="Jobs by Experience Level")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🏢 Company Size Distribution")
        size_counts = df['company_size'].value_counts()
        fig3 = px.pie(values=size_counts.values,
                      names=size_counts.index, title="Jobs by Company Size")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("💼 Salary by Experience Level & Company Size")
    exp_size_salary = df.groupby(['experience_level', 'company_size'])[
        'salary_in_usd'].mean().reset_index()
    fig4 = px.bar(exp_size_salary, x='experience_level', y='salary_in_usd', color='company_size',
                  barmode='group', title="Average Salary by Experience and Company Size",
                  labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Avg Salary (USD)'})
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🔝 Most Common Job Titles")
    top_jobs = df['job_title'].value_counts().head(10)
    fig5 = px.bar(x=top_jobs.values, y=top_jobs.index, orientation='h',
                  title="Top 10 Most Common Job Titles",
                  labels={'x': 'Number of Jobs', 'y': 'Job Title'})
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)
