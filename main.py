import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Insurance Data Analysis",
    page_icon="📊",
    layout="wide"
)

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
tab_tabular, tab_stats, tab_graphical = st.tabs(["Tabular Analysis", "Descriptive Statistics", "Graphical Analysis"])

# Tabular Analysis Tab
with tab_tabular:
    # Display data explorer with a cleaner look
    st.header("Data Explorer")
    st.dataframe(filtered_df, use_container_width=True, height=300)

# Descriptive Statistics Tab
with tab_stats:
    st.header("Descriptive Statistical Measures")
    
    # Create sub-tabs for different statistical views
    stats_tabs = st.tabs(["Summary Statistics", "Numerical Variables", "Categorical Variables", "Aggregated Views", "Correlation Analysis"])
    
    with stats_tabs[0]:  # Summary Statistics
        # Overall dataset statistics
        st.subheader("Dataset Overview")
        st.write(f"Total records: {len(filtered_df)}")
        st.write(f"Number of variables: {filtered_df.shape[1]}")
        
        # Summary statistics for numerical variables
        st.subheader("Summary Statistics for Numerical Variables")
        numerical_stats = filtered_df.describe().T
        # Format the numerical values for better display
        formatted_stats = numerical_stats.copy()
        for col in formatted_stats.columns:
            if col in ['mean', '50%', 'std', 'min', 'max']:
                if col == '50%':
                    formatted_stats.rename(columns={'50%': 'median'}, inplace=True)
                    col = 'median'
                formatted_stats[col] = formatted_stats[col].map(lambda x: f"{x:,.2f}")
                
        st.dataframe(formatted_stats, use_container_width=True)
        
    with stats_tabs[1]:  # Numerical Variables Details
        # Select a numerical variable to analyze
        numerical_cols = ['age', 'bmi', 'children', 'charges']
        selected_num_col = st.selectbox("Select a numerical variable", numerical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display central tendency measures
            st.subheader("Central Tendency")
            central_metrics = {
                "Mean": filtered_df[selected_num_col].mean(),
                "Median": filtered_df[selected_num_col].median(),
                "Mode": filtered_df[selected_num_col].mode()[0]
            }
            
            for metric, value in central_metrics.items():
                st.metric(label=metric, value=f"{value:,.2f}")
                
        with col2:
            # Display dispersion measures
            st.subheader("Dispersion Measures")
            dispersion_metrics = {
                "Standard Deviation": filtered_df[selected_num_col].std(),
                "Variance": filtered_df[selected_num_col].var(),
                "Range": filtered_df[selected_num_col].max() - filtered_df[selected_num_col].min()
            }
            
            for metric, value in dispersion_metrics.items():
                st.metric(label=metric, value=f"{value:,.2f}")
        
        # Show additional metrics
        st.subheader("Additional Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Minimum", f"{filtered_df[selected_num_col].min():,.2f}")
        with col2:
            st.metric("Maximum", f"{filtered_df[selected_num_col].max():,.2f}")
        with col3:
            st.metric("IQR", f"{filtered_df[selected_num_col].quantile(0.75) - filtered_df[selected_num_col].quantile(0.25):,.2f}")

        # Show percentiles
        st.subheader("Percentiles")
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_values = [filtered_df[selected_num_col].quantile(p) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            'Percentile': [f"{int(p*100)}%" for p in percentiles],
            'Value': [f"{v:,.2f}" for v in percentile_values]
        })
        
        st.dataframe(percentile_df, use_container_width=True)
        
    with stats_tabs[2]:  # Categorical Variables
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
        
        st.write(f"Percentage distribution of {second_cat_col} within each {selected_cat_col} category:")
        st.dataframe(formatted_cont_table, use_container_width=True)
        
        # Show raw counts too
        raw_cont_table = pd.crosstab(filtered_df[selected_cat_col], filtered_df[second_cat_col])
        st.write(f"Raw count distribution:")
        st.dataframe(raw_cont_table, use_container_width=True)
        
    with stats_tabs[3]:  # Aggregated Views (Moved from Tabular Analysis)
        st.subheader("Aggregated Data by Categories")
        st.write("Explore how insurance charges vary across different categorical variables.")
        
        group_options = ['sex', 'smoker', 'region', 'age']
        group_by = st.selectbox("Group by", group_options)

        if group_by == 'age':
            # For age, create reasonable bins
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
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
        
        # Additional options for advanced analysis
        st.subheader("Advanced Aggregation")
        
        # Option to add a secondary grouping variable
        sec_group_options = ['None'] + [opt for opt in group_options if opt != group_by]
        secondary_group = st.selectbox("Add secondary grouping", sec_group_options)
        
        if secondary_group != 'None':
            st.write(f"Insurance charges grouped by {group_by.capitalize()} and {secondary_group.capitalize()}:")
            
            if secondary_group == 'age' and group_by != 'age':
                # Handle age binning for secondary grouping
                filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=age_bins, labels=age_labels, right=False)
                pivot_table = pd.pivot_table(
                    filtered_df, 
                    values='charges', 
                    index=group_by, 
                    columns='age_group',
                    aggfunc='mean'
                ).round(2)
                
            elif group_by == 'age' and secondary_group != 'age':
                # Handle age binning for primary grouping
                pivot_table = pd.pivot_table(
                    filtered_df, 
                    values='charges', 
                    index='age_group', 
                    columns=secondary_group,
                    aggfunc='mean'
                ).round(2)
                
            else:
                # Regular pivot table for non-age groupings
                pivot_table = pd.pivot_table(
                    filtered_df, 
                    values='charges', 
                    index=group_by, 
                    columns=secondary_group,
                    aggfunc='mean'
                ).round(2)
            
            # Format as currency
            formatted_pivot = pivot_table.applymap(lambda x: f"${x:,.2f}")
            st.dataframe(formatted_pivot, use_container_width=True)
            
            st.caption("Values represent average insurance charges")
            
    with stats_tabs[4]:  # Correlation Analysis
        # Compute correlation matrix
        numerical_df = filtered_df[['age', 'bmi', 'children', 'charges']]
        corr_matrix = numerical_df.corr().round(3)
        
        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix, use_container_width=True)
        
        # Display interpretation guide
        st.subheader("Interpretation Guide")
        st.write("""
        - **Perfect positive correlation (1.0)**: As one variable increases, the other variable increases by a proportionate amount.
        - **Strong positive correlation (0.7 to 0.9)**: As one variable increases, the other variable tends to increase.
        - **Moderate positive correlation (0.4 to 0.6)**: Some tendency for one variable to increase as the other increases.
        - **Weak positive correlation (0.1 to 0.3)**: Slight tendency for one variable to increase as the other increases.
        - **No correlation (0)**: No relationship between the variables.
        - **Negative correlations**: Similar to positive correlations but in the opposite direction.
        """)
        
        # Show strongest correlations
        st.subheader("Key Relationships")
        
        # Flatten the correlation matrix and find the strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variables': f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            # Display correlations with interpretations
            for _, row in corr_df.iterrows():
                corr_value = row['Correlation']
                vars_pair = row['Variables']
                
                if abs(corr_value) > 0.7:
                    strength = "Strong"
                elif abs(corr_value) > 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                direction = "positive" if corr_value > 0 else "negative"
                
                st.write(f"{vars_pair}: {corr_value:.3f} - {strength} {direction} correlation")

# Graphical Analysis Tab
with tab_graphical:
    st.header("Graphical Analysis")
    st.subheader("Coming Soon!")
    st.write("Interactive visualizations and charts will be available in future updates.")
    st.info("This section will include charts such as distribution plots, correlation matrices, and comparative analyses.")

# Add footer with data source information
st.markdown("---")
st.caption("Data Source: Insurance dataset containing information about policyholders including age, gender, BMI, children, smoker status, region, and charges.")
st.caption("© 2023 Insurance Data Analysis Tool")
