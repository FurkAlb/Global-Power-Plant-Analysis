import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import streamlit as st
import math
import os

# Set page title and layout
st.set_page_config(
    page_title="Energy Data Analysis",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("Global Power Plant Database Analysis")
st.markdown("""
This application performs analysis on the Global Power Plant Database, examining energy generation patterns 
and building predictive models. The analysis includes data cleaning, visualization, feature engineering, 
and a machine learning model to predict energy generation.
""")

# Developer information
st.sidebar.markdown("---")
st.sidebar.subheader("Developer Info")

# Use HTML to create logo-style links
linkedin_logo = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
github_logo = "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" 

st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <img src="{linkedin_logo}" style="height: 20px; margin-right: 10px;">
        <a href="https://www.linkedin.com/in/youssefraafat/" target="_blank">Youssef Raafat</a>
    </div>
    <div style="display: flex; align-items: center;">
        <img src="{github_logo}" style="height: 20px; margin-right: 10px;">
        <a href="https://github.com/youssefraafat" target="_blank">youssefraafat</a>
    </div>
    """, 
    unsafe_allow_html=True
)

# Create sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Data Overview", "Data Cleaning", "Exploratory Analysis", "Feature Engineering", "Model Building", "Model Evaluation"]
selection = st.sidebar.radio("Go to", pages)

# Function to load data
@st.cache_data
def load_data():
    file_path = 'global_power_plant_database.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path, low_memory=False)
    else:
        st.error(f"Could not find {file_path}. Please make sure the file exists in the current directory.")
        return None

# Load the dataset
power_plants = load_data()

if power_plants is not None:
    # Page 1: Data Overview
    if selection == "Data Overview":
        st.header("1. Data Overview")
        
        # Display basic information about the dataset
        st.subheader("Dataset Shape")
        st.write(f"Number of rows: {power_plants.shape[0]}")
        st.write(f"Number of columns: {power_plants.shape[1]}")
        
        # Display the first few rows
        st.subheader("First 5 Rows of Dataset")
        st.dataframe(power_plants.head())
        
        # Display column information
        st.subheader("Column Information")
        buffer = pd.DataFrame({
            'Column Name': power_plants.columns,
            'Data Type': power_plants.dtypes,
            'Non-Null Count': power_plants.count(),
            'Null Count': power_plants.isna().sum(),
            'Null Percentage': (power_plants.isna().sum() / len(power_plants) * 100).round(2)
        })
        st.dataframe(buffer.sort_values('Null Percentage', ascending=False))
        
        # Display countries represented in the dataset
        st.subheader("Countries Represented in Dataset")
        countries_count = power_plants['country'].nunique()
        st.write(f"There are {countries_count} countries in the dataset.")
        
        # Display fuel types represented in the dataset
        st.subheader("Primary Fuel Types")
        primary_fuel_counts = power_plants['primary_fuel'].value_counts().reset_index()
        primary_fuel_counts.columns = ['Fuel Type', 'Count']
        st.dataframe(primary_fuel_counts)
        
        # Interactive visualization of fuel types
        fig = px.pie(primary_fuel_counts, values='Count', names='Fuel Type', 
                     title='Distribution of Power Plants by Primary Fuel Type')
        st.plotly_chart(fig)
    
    # Page 2: Data Cleaning
    elif selection == "Data Cleaning":
        st.header("2. Data Cleaning")
        
        # Check for duplicate rows
        st.subheader("Duplicate Rows")
        duplicate_count = power_plants.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")
        
        # Missing values visualization
        st.subheader("Missing Values Analysis")
        nan_counts = power_plants.isna().sum().sort_values(ascending=False)
        nan_percentage = (nan_counts / len(power_plants) * 100).round(2)
        
        # Display columns with missing values
        missing_data = pd.DataFrame({
            'Missing Values': nan_counts,
            'Percentage (%)': nan_percentage
        }).head(20)
        st.dataframe(missing_data)
        
        # Visualization of missing values
        fig = px.bar(missing_data.reset_index().head(20), 
                     x='index', y='Percentage (%)', 
                     title='Percentage of Missing Values by Column (Top 20)')
        fig.update_xaxes(title="Column Name")
        st.plotly_chart(fig)
        
        # Data cleaning process explanation
        st.subheader("Data Cleaning Process")
        st.markdown("""
        For this analysis, we'll take the following cleaning steps:
        1. Select columns of interest for energy generation analysis
        2. Filter to keep only plants with known capacity
        3. Handle missing values in generation data
        4. Create a clean dataset for modeling
        """)
        
        # Select columns of interest
        columns_to_keep = [
            'country', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel',
            'commissioning_year', 'generation_gwh_2013', 'generation_gwh_2014',
            'generation_gwh_2015', 'generation_gwh_2016', 'generation_gwh_2017', 
            'generation_gwh_2018', 'generation_gwh_2019'
        ]
        
        # Filter to keep only columns we need
        df = power_plants[columns_to_keep].copy()
        
        # Filter to keep only plants with known capacity
        df = df[df['capacity_mw'].notna()]
        
        # Show the cleaned dataset
        st.subheader("Cleaned Dataset (Initial)")
        st.write(f"Cleaned dataset shape: {df.shape}")
        st.dataframe(df.head())
        
        # Calculate the percentage of missing values for generation data
        generation_columns = [col for col in df.columns if 'generation_gwh' in col]
        missing_generation = df[generation_columns].isna().mean() * 100
        
        st.subheader("Missing Generation Data Analysis")
        st.dataframe(pd.DataFrame({
            'Column': missing_generation.index,
            'Missing Percentage (%)': missing_generation.values
        }))
        
        # Creating a year count column for generation data
        df['years_with_data'] = df[generation_columns].notna().sum(axis=1)
        
        # Count plants by number of years with data
        years_data_counts = df['years_with_data'].value_counts().sort_index()
        
        st.subheader("Distribution of Plants by Years with Generation Data")
        fig = px.bar(
            x=years_data_counts.index, 
            y=years_data_counts.values,
            labels={'x': 'Number of Years with Data', 'y': 'Number of Plants'},
            title='Number of Plants by Years with Generation Data'
        )
        st.plotly_chart(fig)
        
        # Final cleaning: plants with at least some generation data
        df_with_gen = df[df['years_with_data'] > 0].copy()
        st.write(f"Plants with at least some generation data: {df_with_gen.shape[0]}")
        
        # For plants with some but not all generation data, fill missing values with the mean of available years
        @st.cache_data
        def fill_missing_generation(df_input):
            df_output = df_input.copy()
            
            def fill_row(row):
                valid_values = []
                
                # Safely extract numeric values
                for col in generation_columns:
                    if pd.notna(row[col]) and pd.api.types.is_numeric_dtype(type(row[col])):
                        valid_values.append(row[col])
                    elif pd.notna(row[col]) and isinstance(row[col], str):
                        try:
                            # Try to convert string to float
                            num_val = float(row[col])
                            valid_values.append(num_val)
                        except ValueError:
                            pass
                            
                if len(valid_values) > 0:
                    mean_gen = np.mean(valid_values)
                    for col in generation_columns:
                        if pd.isna(row[col]):
                            row[col] = mean_gen
                return row
            
            # Apply the function to fill missing generation values
            df_output = df_output.apply(fill_row, axis=1)
            return df_output
            
        df_with_gen = fill_missing_generation(df_with_gen)
        
        # Verify that we've filled missing generation values
        missing_after = df_with_gen[generation_columns].isna().mean() * 100
        
        st.subheader("Missing Values After Filling")
        st.dataframe(pd.DataFrame({
            'Column': missing_after.index,
            'Missing Percentage (%)': missing_after.values
        }))
        
        # Final clean dataset
        st.subheader("Final Cleaned Dataset")
        st.dataframe(df_with_gen.head())
        st.write(f"Final dataset shape: {df_with_gen.shape}")
        
        # Store the clean dataset in session state for later use
        st.session_state['clean_data'] = df_with_gen
    
    # Page 3: Exploratory Analysis
    elif selection == "Exploratory Analysis":
        st.header("3. Exploratory Data Analysis")
        
        # Check if clean data exists in session state
        if 'clean_data' not in st.session_state:
            # If not, create it here (repeat cleaning process)
            columns_to_keep = [
                'country', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel',
                'commissioning_year', 'generation_gwh_2013', 'generation_gwh_2014',
                'generation_gwh_2015', 'generation_gwh_2016', 'generation_gwh_2017', 
                'generation_gwh_2018', 'generation_gwh_2019'
            ]
            
            df = power_plants[columns_to_keep].copy()
            df = df[df['capacity_mw'].notna()]
            
            generation_columns = [col for col in df.columns if 'generation_gwh' in col]
            df['years_with_data'] = df[generation_columns].notna().sum(axis=1)
            df_with_gen = df[df['years_with_data'] > 0].copy()
            
            # Fill missing generation values
            def fill_row(row):
                valid_values = []
                
                # Safely extract numeric values
                for col in generation_columns:
                    if pd.notna(row[col]) and pd.api.types.is_numeric_dtype(type(row[col])):
                        valid_values.append(row[col])
                    elif pd.notna(row[col]) and isinstance(row[col], str):
                        try:
                            # Try to convert string to float
                            num_val = float(row[col])
                            valid_values.append(num_val)
                        except ValueError:
                            pass
                            
                if len(valid_values) > 0:
                    mean_gen = np.mean(valid_values)
                    for col in generation_columns:
                        if pd.isna(row[col]):
                            row[col] = mean_gen
                return row
            
            df_with_gen = df_with_gen.apply(fill_row, axis=1)
            st.session_state['clean_data'] = df_with_gen
        
        df_with_gen = st.session_state['clean_data']
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df_with_gen.describe())
        
        # Distribution of power plants by country
        st.subheader("Distribution by Country")
        top_countries = df_with_gen['country'].value_counts().head(15)
        
        fig = px.bar(
            x=top_countries.index, 
            y=top_countries.values,
            labels={'x': 'Country', 'y': 'Number of Plants'},
            title='Number of Power Plants by Country (Top 15)'
        )
        st.plotly_chart(fig)
        
        # Distribution by fuel type
        st.subheader("Distribution by Fuel Type")
        fuel_counts = df_with_gen['primary_fuel'].value_counts()
        
        fig = px.bar(
            x=fuel_counts.index, 
            y=fuel_counts.values,
            labels={'x': 'Primary Fuel', 'y': 'Number of Plants'},
            title='Number of Power Plants by Primary Fuel Type'
        )
        st.plotly_chart(fig)
        
        # Interactive map
        st.subheader("Global Distribution of Power Plants")
        
        # Sample to prevent overplotting if dataset is very large
        sample_size = min(5000, len(df_with_gen))
        map_sample = df_with_gen.sample(sample_size) if len(df_with_gen) > 5000 else df_with_gen
        
        fig = px.scatter_geo(
            map_sample,
            lat='latitude',
            lon='longitude',
            color='primary_fuel',
            hover_name='name',
            hover_data=['country', 'capacity_mw', 'commissioning_year'],
            title='Global Power Plant Distribution by Fuel Type',
            size='capacity_mw',
            size_max=15,
            opacity=0.7
        )
        
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor='rgb(217, 217, 217)',
                countrycolor='rgb(255, 255, 255)',
                coastlinecolor='rgb(255, 255, 255)',
                showocean=True,
                oceancolor='rgb(230, 230, 250)'
            )
        )
        
        st.plotly_chart(fig)
        
        # Capacity distribution analysis
        st.subheader("Capacity Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_with_gen, 
                x='capacity_mw',
                nbins=50,
                marginal='box',
                title='Distribution of Power Plant Capacity (MW)',
                labels={'capacity_mw': 'Capacity (MW)'}
            )
            fig.update_xaxes(range=[0, df_with_gen['capacity_mw'].quantile(0.99)])
            st.plotly_chart(fig)
        
        with col2:
            # Log transformation for better visualization
            df_with_gen['log_capacity'] = np.log1p(df_with_gen['capacity_mw'])
            
            fig = px.histogram(
                df_with_gen, 
                x='log_capacity',
                nbins=50,
                marginal='box',
                title='Log-Transformed Distribution of Capacity',
                labels={'log_capacity': 'Log(Capacity + 1) (MW)'}
            )
            st.plotly_chart(fig)
        
        # Capacity by fuel type
        st.subheader("Capacity by Fuel Type")
        
        fig = px.box(
            df_with_gen, 
            x='primary_fuel', 
            y='capacity_mw',
            title='Distribution of Power Plant Capacity by Primary Fuel',
            labels={'primary_fuel': 'Primary Fuel', 'capacity_mw': 'Capacity (MW)'}
        )
        fig.update_yaxes(range=[0, df_with_gen['capacity_mw'].quantile(0.95)])
        st.plotly_chart(fig)
        
        # Timeline of plant commissioning
        st.subheader("Historical Trends")
        
        # Filter out null commissioning years and unrealistic values
        valid_years = df_with_gen[
            (df_with_gen['commissioning_year'].notna()) & 
            (df_with_gen['commissioning_year'] >= 1900) & 
            (df_with_gen['commissioning_year'] <= 2023)
        ]
        
        # If we have commissioning year data, show the timeline
        if not valid_years.empty:
            # Group by year and fuel type
            year_fuel_df = valid_years.groupby(['commissioning_year', 'primary_fuel']).size().reset_index(name='count')
            
            fig = px.area(
                year_fuel_df, 
                x='commissioning_year', 
                y='count', 
                color='primary_fuel',
                title='Power Plant Commissioning Timeline by Fuel Type',
                labels={'commissioning_year': 'Commissioning Year', 'count': 'Number of Plants', 'primary_fuel': 'Fuel Type'}
            )
            st.plotly_chart(fig)
            
        # Total capacity by country and fuel type
        st.subheader("Total Capacity by Country and Fuel Type")
        
        # Calculate total capacity by country and fuel type
        country_fuel_capacity = df_with_gen.groupby(['country', 'primary_fuel'])['capacity_mw'].sum().reset_index()
        
        # Get top 15 countries by total capacity
        top_countries_capacity = df_with_gen.groupby('country')['capacity_mw'].sum().nlargest(15).index
        country_fuel_top = country_fuel_capacity[country_fuel_capacity['country'].isin(top_countries_capacity)]
        
        fig = px.bar(
            country_fuel_top,
            x='country', 
            y='capacity_mw', 
            color='primary_fuel',
            title='Total Power Generation Capacity by Country and Fuel Type (Top 15 Countries)',
            labels={'country': 'Country', 'capacity_mw': 'Capacity (MW)', 'primary_fuel': 'Fuel Type'},
            height=600
        )
        
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            legend=dict(title='Fuel Type', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig)
        
        # Correlation between capacity and generation
        st.subheader("Capacity vs. Generation Analysis")
        
        # We'll look at 2019 generation as it's the most recent
        plants_with_2019 = df_with_gen[df_with_gen['generation_gwh_2019'].notna()]
        
        if not plants_with_2019.empty:
            fig = px.scatter(
                plants_with_2019,
                x='capacity_mw', 
                y='generation_gwh_2019',
                color='primary_fuel',
                hover_name='name',
                hover_data=['country', 'capacity_mw', 'commissioning_year'],
                title='Plant Capacity vs. 2019 Generation by Fuel Type',
                labels={'capacity_mw': 'Capacity (MW)', 'generation_gwh_2019': 'Generation in 2019 (GWh)', 'primary_fuel': 'Fuel Type'}
            )
            
            # Add a trend line
            fig.update_layout(
                xaxis=dict(range=[0, plants_with_2019['capacity_mw'].quantile(0.99)]),
                yaxis=dict(range=[0, plants_with_2019['generation_gwh_2019'].quantile(0.99)])
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("No plants with 2019 generation data found.")
        
        # Correlation heatmap
        st.subheader("Correlation Analysis")
        
        # Select numeric columns
        numeric_cols = df_with_gen.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) > 0:
            # Calculate correlation matrix
            corr_matrix = df_with_gen[numeric_cols].corr()
            
            # Plot correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                title='Correlation Heatmap of Numeric Features',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns found for correlation analysis.")
    
    # Page 4: Feature Engineering
    elif selection == "Feature Engineering":
        st.header("4. Feature Engineering")
        
        # Check if clean data exists in session state
        if 'clean_data' not in st.session_state:
            # If not, create it here (repeat cleaning process)
            columns_to_keep = [
                'country', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel',
                'commissioning_year', 'generation_gwh_2013', 'generation_gwh_2014',
                'generation_gwh_2015', 'generation_gwh_2016', 'generation_gwh_2017', 
                'generation_gwh_2018', 'generation_gwh_2019'
            ]
            
            df = power_plants[columns_to_keep].copy()
            df = df[df['capacity_mw'].notna()]
            
            generation_columns = [col for col in df.columns if 'generation_gwh' in col]
            df['years_with_data'] = df[generation_columns].notna().sum(axis=1)
            df_with_gen = df[df['years_with_data'] > 0].copy()
            
            # Fill missing generation values
            def fill_row(row):
                valid_values = []
                
                # Safely extract numeric values
                for col in generation_columns:
                    if pd.notna(row[col]) and pd.api.types.is_numeric_dtype(type(row[col])):
                        valid_values.append(row[col])
                    elif pd.notna(row[col]) and isinstance(row[col], str):
                        try:
                            # Try to convert string to float
                            num_val = float(row[col])
                            valid_values.append(num_val)
                        except ValueError:
                            pass
                            
                if len(valid_values) > 0:
                    mean_gen = np.mean(valid_values)
                    for col in generation_columns:
                        if pd.isna(row[col]):
                            row[col] = mean_gen
                return row
            
            df_with_gen = df_with_gen.apply(fill_row, axis=1)
            st.session_state['clean_data'] = df_with_gen
        
        df_with_gen = st.session_state['clean_data']
        
        st.subheader("Feature Engineering Process")
        st.markdown("""
        We'll create the following engineered features to improve our analysis and modeling:
        1. **Capacity Factor**: Ratio of actual generation to theoretical maximum generation
        2. **Plant Age**: Age of the plant based on commissioning year
        3. **Average Annual Generation**: Mean generation across all available years
        4. **Generation Trend**: Trend in generation over time (increasing/decreasing)
        5. **Generation Variability**: Standard deviation of generation across years
        """)
        
        # Create a new dataframe for feature engineering
        df_features = df_with_gen.copy()
        generation_columns = [col for col in df_features.columns if 'generation_gwh' in col]
        
        # 1. Capacity Factor for each year
        st.subheader("1. Capacity Factor")
        st.markdown("""
        Capacity factor is the ratio of actual electricity generation to the maximum possible generation.
        It indicates how effectively a plant's capacity is being utilized.
        
        Formula: Capacity Factor = Generation (GWh) / (Capacity (MW) × 8760 hours × 0.001)
        """)
        
        # Create capacity factor features
        for year_col in generation_columns:
            year = year_col.split('_')[-1]
            cf_col = f'capacity_factor_{year}'
            
            # Capacity factor = Generation (GWh) / (Capacity (MW) * 8760 hours * 0.001)
            # 8760 is hours in a year, 0.001 converts MWh to GWh
            df_features[cf_col] = df_features[year_col] / (df_features['capacity_mw'] * 8760 * 0.001)
            
            # Cap at 1.0 (theoretical maximum) to handle outliers or data errors
            df_features[cf_col] = df_features[cf_col].clip(upper=1.0)
        
        # Show the new capacity factor columns
        capacity_factor_cols = [col for col in df_features.columns if 'capacity_factor' in col]
        st.dataframe(df_features[['country', 'name', 'primary_fuel', 'capacity_mw'] + capacity_factor_cols].head())
        
        # Visualize capacity factors by fuel type
        mean_cf_by_fuel = df_features.groupby('primary_fuel')[capacity_factor_cols].mean().mean(axis=1).sort_values(ascending=False)
        
        fig = px.bar(
            x=mean_cf_by_fuel.index,
            y=mean_cf_by_fuel.values,
            labels={'x': 'Primary Fuel', 'y': 'Average Capacity Factor'},
            title='Average Capacity Factor by Fuel Type',
            color=mean_cf_by_fuel.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)
        
        # 2. Plant Age
        st.subheader("2. Plant Age")
        
        # Calculate plant age (as of 2023)
        current_year = 2023  # Using 2023 as the reference year
        df_features['plant_age'] = None
        valid_year_mask = (df_features['commissioning_year'].notna()) & (df_features['commissioning_year'] <= current_year)
        df_features.loc[valid_year_mask, 'plant_age'] = current_year - df_features.loc[valid_year_mask, 'commissioning_year']
        
        # Display plants with age information
        st.dataframe(df_features[['country', 'name', 'primary_fuel', 'commissioning_year', 'plant_age']].head())
        
        # Visualize age distribution
        valid_age_data = df_features[df_features['plant_age'].notna()]
        
        if not valid_age_data.empty:
            fig = px.histogram(
                valid_age_data,
                x='plant_age',
                nbins=50,
                color='primary_fuel',
                title='Distribution of Power Plant Ages',
                labels={'plant_age': 'Plant Age (Years)', 'primary_fuel': 'Primary Fuel'}
            )
            st.plotly_chart(fig)
        else:
            st.warning("No plants with valid age data found.")
        
        # 3. Average Annual Generation
        st.subheader("3. Average Annual Generation")
        
        # Calculate average generation across all years
        df_features['avg_annual_generation'] = df_features[generation_columns].mean(axis=1)
        
        # Display average generation
        st.dataframe(df_features[['country', 'name', 'primary_fuel', 'capacity_mw', 'avg_annual_generation']].head())
        
        # Visualize average generation by fuel type
        avg_gen_by_fuel = df_features.groupby('primary_fuel')['avg_annual_generation'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=avg_gen_by_fuel.index,
            y=avg_gen_by_fuel.values,
            labels={'x': 'Primary Fuel', 'y': 'Average Annual Generation (GWh)'},
            title='Average Annual Generation by Fuel Type',
            color=avg_gen_by_fuel.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)
        
        # 4. Generation Trend
        st.subheader("4. Generation Trend")
        
        # Calculate the trend in generation (simplified approach)
        if 'generation_gwh_2013' in df_features.columns and 'generation_gwh_2019' in df_features.columns:
            trend_mask = (df_features['generation_gwh_2013'].notna() & df_features['generation_gwh_2019'].notna() & 
                         (df_features['generation_gwh_2013'] > 0))
            
            df_features['generation_trend'] = df_features['generation_gwh_2019'] - df_features['generation_gwh_2013']
            df_features['generation_trend_pct'] = np.nan  # Initialize with NaN
            
            # Calculate percentage change only where initial generation is > 0
            df_features.loc[trend_mask, 'generation_trend_pct'] = (
                (df_features.loc[trend_mask, 'generation_gwh_2019'] - df_features.loc[trend_mask, 'generation_gwh_2013']) / 
                df_features.loc[trend_mask, 'generation_gwh_2013'] * 100
            )
            
            # Display generation trend
            trend_cols = ['country', 'name', 'primary_fuel', 'generation_gwh_2013', 'generation_gwh_2019', 
                         'generation_trend', 'generation_trend_pct']
            st.dataframe(df_features[trend_cols].head())
            
            # Visualize generation trend by fuel type (for plants with valid trend data)
            valid_trend = df_features[df_features['generation_trend_pct'].notna()]
            
            if not valid_trend.empty:
                trend_by_fuel = valid_trend.groupby('primary_fuel')['generation_trend_pct'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=trend_by_fuel.index,
                    y=trend_by_fuel.values,
                    labels={'x': 'Primary Fuel', 'y': 'Average Generation Trend (% Change 2013-2019)'},
                    title='Average Generation Trend by Fuel Type (2013-2019)',
                    color=trend_by_fuel.values,
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig)
            else:
                st.warning("No plants with valid trend data found.")
        else:
            st.warning("Generation data for 2013 or 2019 is not available for trend calculation.")
        
        # 5. Generation Variability
        st.subheader("5. Generation Variability")
        
        # Calculate variability as coefficient of variation
        # Ensure we have at least two years of data
        variability_mask = df_features[generation_columns].notna().sum(axis=1) >= 2
        df_features['generation_std'] = df_features[generation_columns].std(axis=1, skipna=True)
        
        # Initialize variability with NaN
        df_features['generation_variability'] = np.nan
        
        # Calculate variability only where we have valid mean generation > 0
        valid_var_mask = variability_mask & (df_features['avg_annual_generation'] > 0)
        df_features.loc[valid_var_mask, 'generation_variability'] = (
            df_features.loc[valid_var_mask, 'generation_std'] / 
            df_features.loc[valid_var_mask, 'avg_annual_generation']
        )
        
        # Display generation variability
        var_cols = ['country', 'name', 'primary_fuel', 'avg_annual_generation', 'generation_std', 'generation_variability']
        st.dataframe(df_features[var_cols].head())
        
        # Visualize variability by fuel type
        valid_var_data = df_features[df_features['generation_variability'].notna()]
        
        if not valid_var_data.empty:
            variability_by_fuel = valid_var_data.groupby('primary_fuel')['generation_variability'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=variability_by_fuel.index,
                y=variability_by_fuel.values,
                labels={'x': 'Primary Fuel', 'y': 'Average Generation Variability (CV)'},
                title='Generation Variability by Fuel Type',
                color=variability_by_fuel.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No plants with valid variability data found.")
        
        # Final engineered dataset
        st.subheader("Final Dataset with Engineered Features")
        st.dataframe(df_features.head())
        
        # Store features in session state
        st.session_state['engineered_data'] = df_features
    
    # Page 5: Model Building
    elif selection == "Model Building":
        st.header("5. Model Building")
        
        # Check if engineered data exists
        if 'engineered_data' not in st.session_state:
            st.warning("Please go to the Feature Engineering page first to create the necessary features.")
            st.stop()
        
        df_features = st.session_state['engineered_data']
        
        st.subheader("Predictive Modeling Process")
        st.markdown("""
        We'll build a linear regression model to predict power plant generation based on features:
        
        1. Define target variable and features
        2. Split data into training and testing sets
        3. Train a linear regression model
        4. Evaluate model performance
        """)
        
        # Let user select the target variable (default to 2019 generation)
        target_options = [col for col in df_features.columns if 'generation_gwh' in col]
        target_var = st.selectbox("Select target variable to predict:", target_options, 
                                 index=target_options.index('generation_gwh_2019') if 'generation_gwh_2019' in target_options else 0)
        
        # Select features for the model
        st.subheader("Feature Selection")
        
        potential_features = ['capacity_mw']
        if 'plant_age' in df_features.columns and df_features['plant_age'].notna().sum() > 0:
            potential_features.append('plant_age')
            
        potential_features.extend(['latitude', 'longitude'])
        
        # Add capacity factor features (excluding the year we're trying to predict)
        target_year = target_var.split('_')[-1]
        capacity_factor_cols = [col for col in df_features.columns if 'capacity_factor' in col and target_year not in col]
        potential_features.extend(capacity_factor_cols)
        
        # Add other engineered features
        if 'avg_annual_generation' in df_features.columns and 'generation_gwh' in target_var:
            # Don't include avg_annual_generation if it's highly correlated with target
            # (it likely contains the target in its calculation)
            pass
        else:
            if 'avg_annual_generation' in df_features.columns:
                potential_features.append('avg_annual_generation')
            
        if 'generation_variability' in df_features.columns and df_features['generation_variability'].notna().sum() > 0:
            potential_features.append('generation_variability')
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features for the model:", 
            potential_features,
            default=['capacity_mw'] if 'capacity_mw' in potential_features else []
        )
        
        if not selected_features:
            st.error("Please select at least one feature for the model.")
            st.stop()
        
        # Add categorical features option
        use_fuel_type = st.checkbox("Include fuel type as categorical feature", value=True)
        
        # Prepare data for modeling
        st.subheader("Data Preparation")
        
        # Filter data for modeling (remove rows with missing values in target or selected features)
        model_df = df_features.dropna(subset=[target_var] + selected_features)
        
        if use_fuel_type:
            model_df = model_df.dropna(subset=['primary_fuel'])
        
        # Show modeling dataset size
        st.write(f"Modeling dataset size: {model_df.shape[0]} rows")
        
        # Ensure we have enough data for modeling
        if model_df.shape[0] < 10:
            st.error("Not enough data for modeling after removing missing values. Please select different features.")
            st.stop()
            
        # Prepare features (X) and target (y)
        X = model_df[selected_features].copy()
        
        # One-hot encode categorical features if selected
        if use_fuel_type:
            # Create dummy variables and drop the first one to avoid multicollinearity
            fuel_dummies = pd.get_dummies(model_df['primary_fuel'], prefix='fuel', drop_first=True)
            X = pd.concat([X, fuel_dummies], axis=1)
        
        y = model_df[target_var]
        
        # Display X and y shapes
        st.write(f"Feature matrix shape: {X.shape}")
        st.write(f"Target vector shape: {y.shape}")
        
        # Train-test split
        st.subheader("Train-Test Split")
        
        # Let user choose the test size
        test_size = st.slider("Select test set size (%):", 10, 40, 20) / 100
        
        # Set random state for reproducibility
        random_state = 42
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Testing set: {X_test.shape[0]} samples")
        
        # Train the linear regression model
        st.subheader("Model Training")
        
        # Initialize model
        lr_model = LinearRegression()
        
        # Fit the model
        with st.spinner("Training model..."):
            lr_model.fit(X_train, y_train)
        
        st.success("Model trained successfully!")
        
        # Display model coefficients
        st.subheader("Model Coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': lr_model.coef_
        })
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            coef_df.head(15),  # Top 15 features by coefficient magnitude
            x='Feature',
            y='Coefficient',
            title='Top 15 Feature Coefficients',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig)
        
        # Save model in session state
        model_data = {
            'model': lr_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_name': target_var
        }
        
        st.session_state['model_data'] = model_data
        
        # Optional: Save the model to a file and provide download option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Model to File"):
                model_filename = 'power_plant_generation_model.pkl'
                with open(model_filename, 'wb') as file:
                    pickle.dump({
                        'model': lr_model,
                        'feature_names': X.columns.tolist(),
                        'target_name': target_var
                    }, file)
                st.session_state['model_saved'] = True
                st.success(f"Model saved to {model_filename}")
        
        with col2:
            # Create download button if model is saved
            if 'model_saved' in st.session_state and st.session_state['model_saved']:
                model_file = 'power_plant_generation_model.pkl'
                with open(model_file, 'rb') as f:
                    model_bytes = f.read()
                
                st.download_button(
                    label="Download Saved Model",
                    data=model_bytes,
                    file_name="power_plant_generation_model.pkl",
                    mime="application/octet-stream",
                    help="Download the trained model to use in other applications"
                )
    
    # Page 6: Model Evaluation
    elif selection == "Model Evaluation":
        st.header("6. Model Evaluation")
        
        # Check if model data exists
        if 'model_data' not in st.session_state:
            st.warning("Please go to the Model Building page first to train a model.")
            st.stop()
        
        model_data = st.session_state['model_data']
        model = model_data['model']
        X_train = model_data['X_train'] 
        X_test = model_data['X_test']
        y_train = model_data['y_train']
        y_test = model_data['y_test']
        target_name = model_data['target_name']
        
        st.subheader("Model Performance Metrics")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R²': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R²': r2_score(y_test, y_test_pred)
        }
        
        # Display metrics
        metrics_df = pd.DataFrame({
            'Training Set': train_metrics,
            'Testing Set': test_metrics
        }).transpose()
        
        st.dataframe(metrics_df)
        
        # Interpret R² value
        r2_test = test_metrics['R²']
        st.write(f"**R² Interpretation:** The model explains {r2_test:.2%} of the variance in {target_name}.")
        
        # Visualize predictions vs actual
        st.subheader("Predictions vs. Actual Values")
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred
        })
        
        # Scatter plot of predicted vs actual
        fig = px.scatter(
            results_df,
            x='Actual',
            y='Predicted',
            title=f'Predicted vs. Actual {target_name}',
            labels={'Actual': f'Actual {target_name}', 'Predicted': f'Predicted {target_name}'},
            opacity=0.7
        )
        
        # Add a perfect prediction line
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig)
        
        # Residuals analysis
        st.subheader("Residuals Analysis")
        
        # Calculate residuals
        results_df['Residuals'] = results_df['Actual'] - results_df['Predicted']
        
        # Plot residuals
        fig = px.scatter(
            results_df,
            x='Predicted',
            y='Residuals',
            title='Residuals vs. Predicted Values',
            labels={'Predicted': f'Predicted {target_name}', 'Residuals': 'Residuals'},
            opacity=0.7
        )
        
        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig)
        
        # Histogram of residuals
        fig = px.histogram(
            results_df,
            x='Residuals',
            title='Distribution of Residuals',
            nbins=50
        )
        st.plotly_chart(fig)
        
        # Model interpretation
        st.subheader("Model Interpretation")
        
        # Calculate feature importance based on coefficients
        feature_importance = pd.DataFrame({
            'Feature': model_data['feature_names'],
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        })
        
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            feature_importance.head(15),  # Top 15 features
            x='Feature',
            y='Coefficient',
            title='Feature Importance (Top 15)',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig)
        
        # Add model summary and insights
        st.subheader("Model Summary and Insights")
        
        st.markdown(f"""
        ### Key Insights from the Model:
        
        1. **Model Performance**: The model achieves an R² of {r2_test:.4f} on the test set, indicating it can explain {r2_test:.2%} of the variance in {target_name}.
        
        2. **Most Important Features**: The top features influencing the predictions are {', '.join(feature_importance['Feature'].head(3).tolist())}.
        
        3. **Error Analysis**: The average prediction error (MAE) is {test_metrics['MAE']:.2f} GWh.
        
        ### Potential Applications:
        
        - **Energy Planning**: This model can help estimate future generation from new or existing power plants.
        - **Investment Decisions**: Helps evaluate potential generation output when considering new power plant investments.
        - **Policy Making**: Supports renewable energy transition planning by modeling expected generation.
        
        ### Limitations:
        
        - The model doesn't account for external factors like market conditions or policy changes.
        - Weather and seasonal variations are not captured in the current feature set.
        - The model performs best within the range of the training data.
        """)
else:
    st.error("Failed to load data. Please check the file path and try again.")