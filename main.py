import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration
st.set_page_config(
    page_title="Insurance Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state= "collapsed"
)

# Function to load and display SVG
def render_svg(svg_file):
    with open(svg_file, "r") as f:
        svg_content = f.read()
    
    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/svg+xml;base64,{b64}" style="height:50px; margin-right: 20px;">
        </div>
    """

# Display logo and title in a layout
st.markdown(render_svg("logo.svg"), unsafe_allow_html=True)

# Add page title and description
st.title("Insurance Data Analysis")
st.write("Comprehensive analysis of insurance charges based on various demographic factors.")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('insurance.csv')
    return data

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Move data filters to sidebar
st.sidebar.header("Data Filters")

# Age filter
age_range = st.sidebar.slider(
    "Age Range", 
    min_value=int(df['age'].min()), 
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Gender filter
gender_options = ['All'] + list(df['sex'].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

# Smoker filter
smoker_options = ['All'] + list(df['smoker'].unique())
selected_smoker = st.sidebar.selectbox("Smoker", smoker_options)

# Region filter
region_options = ['All'] + list(df['region'].unique())
selected_region = st.sidebar.selectbox("Region", region_options)

# Filter data based on selections
filtered_df = df.copy()

# Apply age filter
filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

# Apply gender filter
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['sex'] == selected_gender]

# Apply smoker filter
if selected_smoker != 'All':
    filtered_df = filtered_df[filtered_df['smoker'] == selected_smoker]

# Apply region filter
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

# Create tabs for different analyses
tab_tabular, tab_stats, tab_graphical = st.tabs(["Data Overview", "Descriptive Statistics", "Graphical Analysis"])

# Tabular Analysis Tab
with tab_tabular:
    # Create sub-tabs within Data Overview
    overview_tab, data_tab = st.tabs(["Data Explanation", "Tabular Representation"])
    
    # Data Explanation tab
    with overview_tab:
        st.header("Insurance Dataset Overview")
        
        st.markdown("""
        ### Dataset Information
        
        Welcome to our insurance data analysis project! We've compiled this dataset containing information about medical insurance costs in the United States. 
        We've included various demographic factors, health metrics, and the resulting insurance charges for individuals.
        We obtained this complete dataset from Kaggle to help us understand what drives healthcare costs.
        
        ### Data Source
        
        We downloaded the dataset from Kaggle: [Healthcare Insurance](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance).
        It contains records from a medical insurance company that we're analyzing to understand the factors that influence 
        insurance costs. The data represents a sample of policy holders across different regions of the 
        United States that we've selected for our analysis.
        
        ### Variables Description
        
        In our dataset, we're working with these key variables:
        
        * **Age**: Age of the primary beneficiary (in years)
        * **Sex**: Gender of the insurance contractor (female/male)
        * **BMI**: Body Mass Index - a measure of body weight relative to height
        * **Children**: Number of dependents covered by the insurance plan
        * **Smoker**: Whether the beneficiary is a smoker (yes/no)
        * **Region**: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
        * **Charges**: Individual medical costs billed by health insurance (in USD)
        
        ### Research Purpose
        
        In our analysis, we're aiming to identify patterns and factors that significantly affect insurance costs. 
        We believe our findings can be valuable for both insurance providers in risk assessment and for individuals 
        looking to understand their potential costs. We hope you find our insights useful!
        """)
    
    # Tabular Data tab
    with data_tab:
        st.header("Insurance Data Table")
        st.write("Explore the dataset with applied filters below:")
        # Display data explorer with a cleaner look
        st.dataframe(filtered_df, use_container_width=True, height=420)

# Descriptive Statistics Tab
with tab_stats:
    st.header("Descriptive Statistical Measures")
    
    # Create sub-tabs for different statistical views - removed "Summary Statistics"
    stats_tabs = st.tabs(["Numerical Variables", "Categorical Variables", "Aggregated Views"])
    
    with stats_tabs[0]:  # Numerical Variables Details (now the first tab)
        # Select a numerical variable to analyze
        numerical_cols = ['age', 'bmi', 'children', 'charges']
        selected_num_col = st.selectbox("Select a numerical variable", numerical_cols)
        
        # Display variable summary
        st.subheader(f"Analysis of {selected_num_col}")
        
        # Central tendency metrics
        st.write("#### Central Tendency")
        central_cols = st.columns(3)
        central_cols[0].metric("Mean", f"{filtered_df[selected_num_col].mean():,.2f}")
        central_cols[1].metric("Median", f"{filtered_df[selected_num_col].median():,.2f}")
        central_cols[2].metric("Mode", f"{filtered_df[selected_num_col].mode()[0]:,.2f}")
        
        # Dispersion metrics
        st.write("#### Dispersion Measures")
        disp_cols = st.columns(3)
        disp_cols[0].metric("Standard Deviation", f"{filtered_df[selected_num_col].std():,.2f}")
        disp_cols[1].metric("Variance", f"{filtered_df[selected_num_col].var():,.2f}")
        disp_cols[2].metric("Range", f"{filtered_df[selected_num_col].max() - filtered_df[selected_num_col].min():,.2f}")
        
        # Range metrics
        st.write("#### Range Values")
        range_cols = st.columns(3)
        range_cols[0].metric("Minimum", f"{filtered_df[selected_num_col].min():,.2f}")
        range_cols[1].metric("Maximum", f"{filtered_df[selected_num_col].max():,.2f}")
        range_cols[2].metric("IQR", f"{filtered_df[selected_num_col].quantile(0.75) - filtered_df[selected_num_col].quantile(0.25):,.2f}")

        # Percentiles
        st.write("#### Percentiles")
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_values = [filtered_df[selected_num_col].quantile(p) for p in percentiles]
        
        # Display percentiles in columns
        perc_cols = st.columns(len(percentiles))
        for i, (p, v) in enumerate(zip(percentiles, percentile_values)):
            perc_cols[i].metric(f"{int(p*100)}th", f"{v:,.2f}")
        
            
    with stats_tabs[1]:  # Categorical Variables (now the second tab)
        # Select a categorical variable
        cat_cols = ['sex', 'smoker', 'region']
        selected_cat_col = st.selectbox("Select a categorical variable", cat_cols)
        
        # Show frequency distribution
        st.subheader("Frequency Distribution")
        cat_counts = filtered_df[selected_cat_col].value_counts().reset_index()
        cat_counts.columns = [selected_cat_col, 'Count']
        
        # Calculate percentages
        total = cat_counts['Count'].sum()
        cat_counts['Percentage'] = (cat_counts['Count'] / total * 100).round(2).astype(str) + '%'
        
        # Display as a table
        st.dataframe(cat_counts, use_container_width=True)
        
        # Contingency tables
        st.subheader("Contingency Tables")
        
        # Select a second categorical variable to create a cross-tabulation
        other_cat_cols = [col for col in cat_cols if col != selected_cat_col]
        second_cat_col = st.selectbox("Select a second categorical variable for cross-tabulation", other_cat_cols)
        
        # Create and display the contingency table
        cont_table = pd.crosstab(
            filtered_df[selected_cat_col], 
            filtered_df[second_cat_col],
            normalize='index'
        ).round(3) * 100
        
        # Format as percentages for display
        formatted_cont_table = cont_table.applymap(lambda x: f"{x:.1f}%")
        
        # Get raw counts too
        raw_cont_table = pd.crosstab(filtered_df[selected_cat_col], filtered_df[second_cat_col])
        
        st.write(f"Distribution of {second_cat_col} within each {selected_cat_col} category:")
        
        # Display both tables together
        col1, col2 = st.columns(2)
        with col1:
            st.write("Percentage Distribution:")
            st.dataframe(formatted_cont_table, use_container_width=True)
        
        with col2:
            st.write("Count Distribution:")
            st.dataframe(raw_cont_table, use_container_width=True)
        
    with stats_tabs[2]:  # Aggregated Views (now the third tab)
        st.subheader("Aggregated Data by Categories")
        st.write("Explore how insurance charges vary across different categorical variables.")
        
        group_options = ['sex', 'smoker', 'region', 'age']
        group_by = st.selectbox("Group by", group_options)

        if group_by == 'age':
            # For age, create reasonable bins
            age_bins = [18, 25, 35, 45, 55, 65]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64']
            filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=age_bins, labels=age_labels, right=False)
            group_data = filtered_df.groupby('age_group')['charges'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
            group_data = group_data.rename(columns={
                'age_group': 'Age Group',
                'mean': 'Mean Charges',
                'median': 'Median Charges',
                'min': 'Min Charges',
                'max': 'Max Charges',
                'count': 'Count'
            })
        else:
            # For other categorical variables
            group_data = filtered_df.groupby(group_by)['charges'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
            group_data = group_data.rename(columns={
                group_by: group_by.capitalize(),
                'mean': 'Mean Charges',
                'median': 'Median Charges',
                'min': 'Min Charges',
                'max': 'Max Charges',
                'count': 'Count'
            })
        
        # Format currency columns
        for col in ['Mean Charges', 'Median Charges', 'Min Charges', 'Max Charges']:
            group_data[col] = group_data[col].map('${:,.2f}'.format)
        
        # Add visual cues
        st.write("Statistical summary of insurance charges grouped by " + group_by.capitalize() + ":")
        st.dataframe(group_data, use_container_width=True)
        
        # Add some insights below the table
        if group_by == 'smoker':
            if 'yes' in group_data[group_by.capitalize()].values and 'no' in group_data[group_by.capitalize()].values:
                smoker_row = group_data[group_data[group_by.capitalize()] == 'yes']
                non_smoker_row = group_data[group_data[group_by.capitalize()] == 'no']
                if not smoker_row.empty and not non_smoker_row.empty:
                    smoker_mean = smoker_row['Mean Charges'].iloc[0]
                    non_smoker_mean = non_smoker_row['Mean Charges'].iloc[0]
                    st.info(f"💡 Insight: Smokers on average have higher insurance charges ({smoker_mean}) compared to non-smokers ({non_smoker_mean}).")

# Graphical Analysis Tab
with tab_graphical:
    st.header("Graphical Analysis")
    st.subheader("Coming Soon!")
    st.write("Interactive visualizations and charts will be available in future updates.")
    st.info("This section will include charts such as distribution plots, correlation matrices, and comparative analyses.")


