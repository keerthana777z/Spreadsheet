import streamlit as st

# 🔧 Layout Configuration
st.set_page_config(page_title="Indian Startup Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor

# --- Load Data ---
df = pd.read_csv('cleaned_startup_funding.csv', encoding='utf-8-sig')

df['City'] = df['City'].str.strip().replace({
    'Gurgaon': 'Gurugram',
    'Gurugram': 'Gurugram',
    'Gurgaon / Sfo': 'Gurugram'
})


df['Investment_Type'] = df['Investment_Type'].str.strip().replace({
    'Seed/ Angel Funding': 'Seed / Angel Funding',
    'Seed / Angel Funding': 'Seed / Angel Funding',
    'Seed/Angel Funding': 'Seed / Angel Funding',
    'Angel / Seed Funding': 'Seed / Angel Funding'
})


# Normalize city names
df['City'] = df['City'].str.strip().replace({
    'Bangalore': 'Bengaluru', 'bangalore': 'Bengaluru',
    'BENGALURU': 'Bengaluru', 'Bengalore': 'Bengaluru'
})
# 🧹 Normalize Industry Names
df['Industry_Vertical'] = df['Industry_Vertical'].str.strip().replace({
    'Ecommerce': 'E-Commerce',
    'ecommerce': 'E-Commerce',
    'E-commerce': 'E-Commerce'
})
# Remove 'Not Disclosed' industry entries
df = df[df['Industry_Vertical'].str.lower() != 'not disclosed']

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'Amount', 'City', 'Industry_Vertical', 'Investment_Type'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df = df.dropna(subset=['Amount'])
df['Year'] = df['Date'].dt.year

valid_df = df.groupby(['Year', 'Industry_Vertical', 'City']).filter(lambda x: len(x) >= 3)
# SAVE A COPY BEFORE ENCODING
df_original = df.copy()


# --- Sidebar ---
st.sidebar.title("🔍 Filters")

if 'Investor_Count' not in valid_df.columns:
    valid_df['Investor_Count'] = valid_df['Investors'].fillna("").apply(lambda x: len(x.split(',')))
valid_df['Investor_Count'] = pd.to_numeric(valid_df['Investor_Count'], errors='coerce')
valid_df['Amount'] = pd.to_numeric(valid_df['Amount'], errors='coerce')
plot_df = valid_df.dropna(subset=['Investor_Count', 'Amount'])

years = sorted(valid_df['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

industries = sorted(valid_df[valid_df['Year'] == selected_year]['Industry_Vertical'].unique())
selected_industry = st.sidebar.selectbox("Select Industry", industries)

cities = sorted(valid_df[
    (valid_df['Year'] == selected_year) & 
    (valid_df['Industry_Vertical'] == selected_industry)
]['City'].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

top_n_startups = st.sidebar.slider("Top N Startups", 3, 20, 10)
# top_n_cities = st.sidebar.slider("Top N Cities", 3, 20, 10)
# top_n_industries = st.sidebar.slider("Top N Industries", 3, 20, 10)
top_n_investments = st.sidebar.slider("Top N Investment Types", 3, 10, 5)

# --- Filtered Data ---
filtered_df = valid_df[
    (valid_df['Year'] == selected_year) & 
    (valid_df['Industry_Vertical'] == selected_industry) &
    (valid_df['City'] == selected_city)
]

# --- Dashboard Title ---
st.title("🇮🇳 Indian Startup Funding Dashboard")

# --- TABS for Compact Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔮 Predictions", "📁 Raw Data"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Funding Trend (Rolling Average)")
        trend_df = filtered_df.copy()
        trend_df = trend_df.set_index('Date').resample('M')['Amount'].sum().rolling(3).mean().reset_index()
        fig = px.line(trend_df, x='Date', y='Amount', title="3-Month Rolling Avg of Funding")
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"💰 Top {top_n_investments} Investment Types (Proportion View)")
        investment_counts = valid_df['Investment_Type'].value_counts().nlargest(top_n_investments)
        fig = px.pie(
            names=investment_counts.index,
            values=investment_counts.values,
            hole=0.5,
            title="Top Investment Types by Count",
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"🏆 Top {top_n_startups} Funded Startups")
        top_startups = valid_df.groupby('Startup')['Amount'].sum().nlargest(top_n_startups)
        fig = px.bar(
        top_startups, 
        x=top_startups.index, 
        y=top_startups.values, 
        labels={'y': 'Funding Amount', 'x': 'Startup'}  # Set custom labels for axes
        )
        fig.update_yaxes(title='Funding Amount')  # Explicitly set the y-axis title
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏙 City Contribution to Each Industry")

        city_industry_df = valid_df.groupby(['Industry_Vertical', 'City'])['Amount'].sum().reset_index()

        top_industries = city_industry_df.groupby('Industry_Vertical')['Amount'].sum().nlargest(7).index
        city_industry_df = city_industry_df[city_industry_df['Industry_Vertical'].isin(top_industries)]

# Add a new column for formatted text
        def format_amount(val):
             if val >= 1e9:
                return f"{val/1e9:.1f}B"
             elif val >= 1e6:
                  return f"{val/1e6:.0f}M"
             else:
                   return f"{val:.0f}"

        city_industry_df['Amount_Label'] = city_industry_df['Amount'].apply(format_amount)

        fig = px.bar(
          city_industry_df,
          x='Industry_Vertical',
          y='Amount',
          color='City',
          title='Funding by Industry Stacked by City',
          text='Amount_Label'  # Use the new formatted column
        )

        fig.update_layout(barmode='stack', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("📊 Interactive Area Chart: Top 5 Industries Over Time")

            # --- Prepare Data ---
        area_df = valid_df.copy()
        area_df['Year'] = area_df['Date'].dt.year

        top_industries = area_df.groupby('Industry_Vertical')['Amount'].sum().nlargest(5).index
        area_df = area_df[area_df['Industry_Vertical'].isin(top_industries)]

        area_grouped = area_df.groupby(['Year', 'Industry_Vertical'])['Amount'].sum().reset_index()

        # --- Plotly Area Chart ---
        fig = px.area(
            area_grouped,
            x='Year',
            y='Amount',
            color='Industry_Vertical',
            title='📈 Interactive Funding Trends (Top 5 Industries)',
            labels={'Amount': 'Total Funding (Rs.)'},
            line_group='Industry_Vertical'
        )
        fig.update_traces(mode='lines+markers', line=dict(width=2))
        fig.update_layout(
            xaxis=dict(title='Year'),
            yaxis=dict(title='Funding Amount'),
            legend_title_text='Industry',
            hovermode="x unified",
            margin=dict(t=50, l=30, r=30, b=30),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)





        st.subheader("📊 Industry Funding Insight")
        viz_type = st.radio("Choose View:", ["Sunburst", "Treemap"], key="viz_radio")
        top_industry_limit = 5
        top_industries = valid_df.groupby('Industry_Vertical')['Amount'].sum().nlargest(top_industry_limit).index
        viz_df = valid_df.copy()
        viz_df['Industry_Vertical'] = viz_df['Industry_Vertical'].apply(lambda x: x if x in top_industries else 'Other')
        viz_df = viz_df[viz_df['Industry_Vertical'] != 'Other']
        if viz_type == "Sunburst":
            fig = px.sunburst(
                viz_df,
                path=['Industry_Vertical', 'Investment_Type', 'City'],
                values='Amount',
                color='Industry_Vertical',
                title=f"Funding Distribution: Top {top_industry_limit} Industries",
                height=500,
            )
            fig.update_traces(insidetextorientation='radial')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.treemap(
                viz_df,
                path=['Industry_Vertical', 'Investment_Type', 'City'],
                values='Amount',
                color='Amount',
                title=f"Funding Breakdown (Treemap) - Top {top_industry_limit} Industries",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌆 Top 7 Cities by Total Funding + Deal Count + Avg Funding")
    top_cities = valid_df.groupby('City')['Amount'].sum().nlargest(7).index
    city_insight_df = valid_df[valid_df['City'].isin(top_cities)]
    city_grouped = city_insight_df.groupby('City').agg({
        'Amount': ['sum', 'count', 'mean']
    }).reset_index()
    city_grouped.columns = ['City', 'Total_Funding', 'Deal_Count', 'Avg_Funding']
    city_grouped = city_grouped.sort_values(by='Total_Funding', ascending=True)
    fig = px.bar(
        city_grouped,
        y='City',
        x='Total_Funding',
        color='Avg_Funding',
        orientation='h',
        text='Deal_Count',
        color_continuous_scale='Agsunset',
        title='Top 7 Cities by Total Funding (Color: Avg Funding, Label: Deal Count)'
    )
    fig.update_layout(
        xaxis_title="Total Funding (Rs)",
        yaxis_title="City",
        coloraxis_colorbar=dict(title="Avg Funding"),
    )
    fig.update_traces(texttemplate='Deals: %{text}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    st.subheader("🌐 Conceptual Word Cloud from Startups & Sectors ")

    # Combine text columns for richer word base
    text_columns = ['Startup', 'SubVertical']
    word_df = filtered_df[text_columns].fillna("").astype(str)
    combined_text = " ".join(word_df[col].str.cat(sep=" ") for col in text_columns)

    # Optional: Additional stopwords
    custom_stopwords = set(STOPWORDS).union({
        'Private', 'Ltd', 'India', 'Technologies', 'Technology', 'Services',
        'Solutions', 'Pvt', 'Limited', 'based', 'online', 'Platform'
    })

    # Generate Word Cloud
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        stopwords=custom_stopwords,
        colormap='tab10',
        max_words=150,
        contour_color='steelblue'
    ).generate(combined_text)

    # Display
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

with tab2:
    st.header("🔮 Predict Startup Funding Amount")
    with st.form("prediction_form"):
        # Input fields for prediction
        pred_year = st.number_input("Year (Enter future year)", min_value=df['Year'].max() + 1, value=df['Year'].max() + 1)
        pred_industry = st.selectbox("Industry", sorted(df['Industry_Vertical'].unique()))
        pred_city = st.selectbox("City", sorted(df['City'].unique()))
        pred_invest = st.selectbox("Investment Type", sorted(df['Investment_Type'].unique()))
        submitted = st.form_submit_button("Predict Amount")

        if submitted:
            try:
                # Data preprocessing
                ml_df = df[['Year', 'Industry_Vertical', 'City', 'Investment_Type', 'Amount']].copy()
                
                # Calculate multipliers based on historical data
                city_multiplier = df[df['City'] == pred_city]['Amount'].mean() / df['Amount'].mean()
                industry_multiplier = df[df['Industry_Vertical'] == pred_industry]['Amount'].mean() / df['Amount'].mean()
                
                # Handle cases where multipliers are NaN or 0
                city_multiplier = 1.0 if pd.isna(city_multiplier) or city_multiplier == 0 else city_multiplier
                industry_multiplier = 1.0 if pd.isna(industry_multiplier) or industry_multiplier == 0 else industry_multiplier

                # Calculate years difference
                years_diff = pred_year - df['Year'].max()
                
                # Calculate base prediction using Gradient Boosting
                encoders = {}
                for col in ['Industry_Vertical', 'City', 'Investment_Type']:
                    enc = LabelEncoder()
                    ml_df[col] = enc.fit_transform(ml_df[col])
                    encoders[col] = enc

                X = ml_df.drop('Amount', axis=1)
                y = ml_df['Amount']
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X, y)

                input_data = pd.DataFrame([{
                    'Year': pred_year,
                    'Industry_Vertical': encoders['Industry_Vertical'].transform([pred_industry])[0],
                    'City': encoders['City'].transform([pred_city])[0],
                    'Investment_Type': encoders['Investment_Type'].transform([pred_invest])[0]
                }])

                base_prediction = model.predict(input_data)[0]
                
                # Apply a more conservative year adjustment
                # Using a dampened growth rate that decreases over time
                yearly_growth_rate = 0.15  # 15% base growth rate
                damping_factor = 0.9  # Reduces growth rate each year
                year_multiplier = 1.0
                
                for year in range(years_diff):
                    current_growth = yearly_growth_rate * (damping_factor ** year)
                    year_multiplier *= (1 + current_growth)

                # Apply multipliers to base prediction
                final_prediction = base_prediction * city_multiplier * industry_multiplier * year_multiplier

                # Ensure prediction is not negative
                final_prediction = max(final_prediction, 0)

                # Display only the final prediction
                st.success(f"💸 Predicted Funding Amount for {pred_year}: Rs.{final_prediction:,.2f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")




with tab3:
    st.subheader("📁 View Raw Data")
    st.dataframe(valid_df)