import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# Set page configuration
st.set_page_config(
    page_title="Punjab Crop Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state= "expanded",
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
st.title("Punjab Crop Data Analysis")
st.write("Comprehensive analysis of crop production and yield across districts in Punjab.")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('croppunjab.csv')
    return data

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Move data filters to sidebar
st.sidebar.header("Data Filters")

# Year range filter
year_min = int(df['Crop_Year'].min())
year_max = int(df['Crop_Year'].max())
year_range = st.sidebar.slider(
    "Year Range", 
    min_value=year_min, 
    max_value=year_max,
    value=(year_min, year_max)
)

# District filter
district_options = ['All'] + sorted(list(df['District'].unique()))
selected_district = st.sidebar.selectbox("District", district_options)

# Crop filter
crop_options = ['All'] + sorted(list(df['Crop'].unique()))
selected_crop = st.sidebar.selectbox("Crop", crop_options)

# Yield range filter
yield_min = float(df['Yield'].min())
yield_max = float(df['Yield'].max())
yield_range = st.sidebar.slider(
    "Yield Range (tons/hectare)", 
    min_value=yield_min, 
    max_value=yield_max,
    value=(yield_min, yield_max),
    step=0.1
)

# Filter data based on selections
filtered_df = df.copy()

# Apply year filter
filtered_df = filtered_df[(filtered_df['Crop_Year'] >= year_range[0]) & (filtered_df['Crop_Year'] <= year_range[1])]

# Apply district filter
if selected_district != 'All':
    filtered_df = filtered_df[filtered_df['District'] == selected_district]

# Apply crop filter
if selected_crop != 'All':
    filtered_df = filtered_df[filtered_df['Crop'] == selected_crop]

# Apply yield filter
filtered_df = filtered_df[(filtered_df['Yield'] >= yield_range[0]) & (filtered_df['Yield'] <= yield_range[1])]

# Create tabs for different analyses
tab_tabular, tab_stats, tab_graphical, tab_probability, tab_regression = st.tabs([
    "Data Overview", "Descriptive Statistics", "Graphical Analysis", 
    "Probability Methods", "Regression Modeling"
])

# Tabular Analysis Tab
with tab_tabular:
    # Create sub-tabs within Data Overview
    overview_tab, data_tab = st.tabs(["Data Explanation", "Tabular Representation"])
    
    # Data Explanation tab
    with overview_tab:
        st.header("Punjab Crop Dataset Overview")
        
        st.markdown("""
        ### Dataset Information
        
        Welcome to our Punjab crop analysis project! We've compiled this dataset containing information about crop production in Punjab, India.
        The dataset includes details about rice and wheat production across different districts over multiple years.
        
        ### Data Source
        
        This dataset contains agricultural statistics from Punjab, focusing on the production metrics and yield of major crops across different districts.
        The data represents official agricultural statistics collected over several years.
        
        ### Variables Description
        
        In our dataset, we're working with these key variables:
        
        * **District**: Administrative district in Punjab
        * **Crop**: Type of crop (Rice or Wheat)
        * **Crop_Year**: Year of crop harvest
        * **Area**: Land area used for cultivation (in hectares)
        * **Production**: Total crop production (in tonnes)
        * **Yield**: Production per unit area (tonnes/hectare)
        
        ### Research Purpose
        
        In our analysis, we aim to identify patterns and factors that significantly affect crop production and yield across different districts.
        We believe our findings can be valuable for agricultural planning, resource allocation, and understanding production trends over time.
        """)
    
    # Tabular Data tab
    with data_tab:
        st.header("Punjab Crop Data Table")
        st.write("Explore the dataset with applied filters below:")
        # Display data explorer with a cleaner look - remove height parameter to avoid PyArrow issues
        st.dataframe(filtered_df, use_container_width=True)

# Descriptive Statistics Tab
with tab_stats:
    st.header("Descriptive Statistical Measures")
    
    # Create sub-tabs for different statistical views
    stats_tabs = st.tabs(["Numerical Variables", "Categorical Variables", "Aggregated Views"])
    
    with stats_tabs[0]:  # Numerical Variables Details
        # Select a numerical variable to analyze
        numerical_cols = ['Area', 'Production', 'Yield']  # Removed Crop_Year
        selected_num_col = st.selectbox("Select a numerical variable", numerical_cols)
        
        # Display variable summary
        st.subheader(f"Analysis of {selected_num_col}")
        
        # Central tendency metrics
        st.write("#### Central Tendency")
        central_cols = st.columns(3)
        central_cols[0].metric("Mean", f"{filtered_df[selected_num_col].mean():,.2f}")
        central_cols[1].metric("Median", f"{filtered_df[selected_num_col].median():,.2f}")
        central_cols[2].metric("Mode", f"{filtered_df[selected_num_col].mode()[0]:,.2f}")
        
        # Add insight for central tendency
        mean_val = filtered_df[selected_num_col].mean()
        median_val = filtered_df[selected_num_col].median()
        skew_insight = "relatively symmetric" if abs(mean_val - median_val) < (mean_val * 0.1) else "right-skewed" if mean_val > median_val else "left-skewed"
        
        st.info(f"""
        **📊 Central Tendency Insight:**
        
        The average {selected_num_col.lower()} is {mean_val:,.2f}, with half of all values falling below {median_val:,.2f} (median).
        
        The distribution appears to be {skew_insight}. {
        "Values are fairly evenly distributed around the average." if skew_insight == "relatively symmetric" else
        "There are likely some unusually high values pulling the average up." if skew_insight == "right-skewed" else
        "There are likely some unusually low values pulling the average down."
        }
        """)
        
        # Dispersion metrics
        st.write("#### Dispersion Measures")
        disp_cols = st.columns(3)
        disp_cols[0].metric("Standard Deviation", f"{filtered_df[selected_num_col].std():,.2f}")
        disp_cols[1].metric("Variance", f"{filtered_df[selected_num_col].var():,.2f}")
        disp_cols[2].metric("Range", f"{filtered_df[selected_num_col].max() - filtered_df[selected_num_col].min():,.2f}")
        
        # Add insight for dispersion
        std_dev = filtered_df[selected_num_col].std()
        cv = (std_dev / mean_val) * 100 if mean_val != 0 else 0
        
        st.info(f"""
        **📏 Dispersion Insight:**
        
        The standard deviation of {std_dev:,.2f} shows how much {selected_num_col.lower()} typically varies from the average.
        
        The coefficient of variation is {cv:.1f}%, indicating {"high" if cv > 30 else "moderate" if cv > 15 else "low"} variability 
        relative to the mean. {
        "This suggests significant differences across the data." if cv > 30 else
        "This suggests moderate differences across the data." if cv > 15 else
        "This suggests relatively consistent values across the data."
        }
        """)
        
        # Range metrics
        st.write("#### Range Values")
        range_cols = st.columns(3)
        range_cols[0].metric("Minimum", f"{filtered_df[selected_num_col].min():,.2f}")
        range_cols[1].metric("Maximum", f"{filtered_df[selected_num_col].max():,.2f}")
        range_cols[2].metric("IQR", f"{filtered_df[selected_num_col].quantile(0.75) - filtered_df[selected_num_col].quantile(0.25):,.2f}")

        # Add insight for range values
        min_val = filtered_df[selected_num_col].min()
        max_val = filtered_df[selected_num_col].max()
        iqr = filtered_df[selected_num_col].quantile(0.75) - filtered_df[selected_num_col].quantile(0.25)
        spread_ratio = (max_val - min_val) / iqr if iqr != 0 else 0
        
        st.info(f"""
        **🔍 Range Insight:**
        
        The values range from {min_val:,.2f} to {max_val:,.2f}, spanning a total of {max_val - min_val:,.2f} units.
        
        The IQR (middle 50% of data) is {iqr:,.2f}, meaning most values are concentrated within this range.
        
        The ratio of total range to IQR is {spread_ratio:.1f}, which {"suggests potential outliers" if spread_ratio > 3 else "indicates a reasonable distribution without extreme outliers"}.
        """)

        # Percentiles
        st.write("#### Percentiles")
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_values = [filtered_df[selected_num_col].quantile(p) for p in percentiles]
        
        # Display percentiles in columns
        perc_cols = st.columns(len(percentiles))
        for i, (p, v) in enumerate(zip(percentiles, percentile_values)):
            perc_cols[i].metric(f"{int(p*100)}th", f"{v:,.2f}")
        
        # Add insight for percentiles
        p90_p10_ratio = percentile_values[4] / percentile_values[0] if percentile_values[0] != 0 else 0
        
        st.info(f"""
        **📈 Percentile Insight:**
        
        The median (50th percentile) is {percentile_values[2]:,.2f}, with 90% of all values falling below {percentile_values[4]:,.2f}.
        
        The ratio between the 90th and 10th percentiles is {p90_p10_ratio:.1f}x, which {"indicates high inequality across observations" if p90_p10_ratio > 5 else "suggests moderate differences" if p90_p10_ratio > 2 else "shows relatively consistent values"}.
        
        {
        f"The top 1% of values exceed {percentile_values[6]:,.2f}, which may represent exceptional cases worthy of special attention." if p90_p10_ratio > 2 else
        "The distribution appears relatively balanced across different percentiles."
        }
        """)
            
    with stats_tabs[1]:  # Categorical Variables
        # Select a categorical variable
        cat_cols = ['District', 'Crop', 'Crop_Year']  # Added Crop_Year here
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
        
        # Add insight for frequency distribution
        top_category = cat_counts.iloc[0][selected_cat_col]
        top_percentage = float(cat_counts.iloc[0]['Percentage'].rstrip('%'))
        n_categories = len(cat_counts)
        hhi_index = sum((float(p.rstrip('%'))/100)**2 for p in cat_counts['Percentage'])
        
        st.info(f"""
        **📊 Frequency Distribution Insight:**
        
        There are {n_categories} unique values for {selected_cat_col}, with "{top_category}" being the most common ({top_percentage:.1f}% of all records).
        
        The Herfindahl-Hirschman Index (measuring concentration) is {hhi_index:.3f}, indicating {"high concentration" if hhi_index > 0.25 else "moderate concentration" if hhi_index > 0.15 else "low concentration"}.
        
        {
        "This suggests that a few categories dominate the dataset." if hhi_index > 0.25 else
        "The distribution has some dominant categories but maintains diversity." if hhi_index > 0.15 else
        "The distribution is relatively balanced across many categories."
        }
        """)
        
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
        
        # Add insight for contingency tables
        max_diff = cont_table.max(axis=1).max() - cont_table.min(axis=1).min()
        
        st.info(f"""
        **🔄 Contingency Table Insight:**
        
        This table shows how {second_cat_col} is distributed within each {selected_cat_col} category.
        
        The maximum difference in distribution percentages is {max_diff:.1f}%, which {"indicates strong associations between the variables" if max_diff > 50 else "suggests moderate associations" if max_diff > 25 else "suggests relatively weak associations"}.
        
        {
        "Look for rows with significantly different distributions - these indicate categories with unique relationships to " + second_cat_col + "." if max_diff > 25 else
        "The distributions appear relatively similar across categories, indicating limited association between these variables."
        }
        """)
        
    with stats_tabs[2]:  # Aggregated Views
        st.subheader("Aggregated Data by Categories")
        st.write("Explore how crop yields vary across different categorical variables.")
        
        group_options = ['District', 'Crop', 'Crop_Year']
        group_by = st.selectbox("Group by", group_options)
        
        # Column name to access in the dataframe after possible renaming
        group_by_col = 'Year' if group_by == 'Crop_Year' else group_by

        if group_by == 'Crop_Year':
            # For years, create reasonable bins
            group_data = filtered_df.groupby('Crop_Year')['Yield'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
            group_data = group_data.rename(columns={
                'Crop_Year': 'Year',
                'mean': 'Mean Yield',
                'median': 'Median Yield',
                'min': 'Min Yield',
                'max': 'Max Yield',
                'count': 'Count'
            })
        else:
            # For other categorical variables
            group_data = filtered_df.groupby(group_by)['Yield'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
            group_data = group_data.rename(columns={
                group_by: group_by,
                'mean': 'Mean Yield',
                'median': 'Median Yield',
                'min': 'Min Yield',
                'max': 'Max Yield',
                'count': 'Count'
            })
        
        # Format columns
        for col in ['Mean Yield', 'Median Yield', 'Min Yield', 'Max Yield']:
            group_data[col] = group_data[col].map('{:,.2f}'.format)
        
        # Add visual cues
        st.write("Statistical summary of crop yields grouped by " + group_by + ":")
        st.dataframe(group_data, use_container_width=True)
        
        # Convert formatted columns back to numeric for calculations
        numeric_group_data = group_data.copy()
        for col in ['Mean Yield', 'Median Yield', 'Min Yield', 'Max Yield']:
            numeric_group_data[col] = numeric_group_data[col].str.replace(',', '').astype(float)
        
        # Calculate metrics for insights
        max_mean_category = numeric_group_data.loc[numeric_group_data['Mean Yield'].idxmax()][group_by_col]
        min_mean_category = numeric_group_data.loc[numeric_group_data['Mean Yield'].idxmin()][group_by_col]
        max_mean_value = numeric_group_data['Mean Yield'].max()
        min_mean_value = numeric_group_data['Mean Yield'].min()
        relative_difference = ((max_mean_value - min_mean_value) / min_mean_value) * 100
        
        # Calculate variability within categories
        highest_range_category = numeric_group_data.loc[(numeric_group_data['Max Yield'] - numeric_group_data['Min Yield']).idxmax()][group_by_col]
        highest_range_value = numeric_group_data.loc[(numeric_group_data['Max Yield'] - numeric_group_data['Min Yield']).idxmax()]
        highest_range = float(highest_range_value['Max Yield']) - float(highest_range_value['Min Yield'])
        
        # Add comprehensive insight based on the group_by variable
        if group_by == 'District':
            st.info(f"""
            **🌾 District Yield Insight:**
            
            The highest average yield is found in {max_mean_category} ({max_mean_value:.2f} tonnes/hectare), while the lowest is in {min_mean_category} ({min_mean_value:.2f} tonnes/hectare).
            
            This represents a {relative_difference:.1f}% difference between the best and worst performing districts.
            
            {highest_range_category} shows the greatest variability in yields, with a range of {highest_range:.2f} tonnes/hectare between minimum and maximum values. This could indicate greater sensitivity to seasonal conditions or varied farming practices.
            
            Districts with consistently high median values represent areas with more stable productivity, potentially indicating better agricultural infrastructure or more favorable growing conditions.
            """)
        elif group_by == 'Crop':
            st.info(f"""
            **🌱 Crop Yield Insight:**
            
            {max_mean_category} shows the highest average yield at {max_mean_value:.2f} tonnes/hectare, while {min_mean_category} has the lowest at {min_mean_value:.2f} tonnes/hectare.
            
            The {relative_difference:.1f}% yield difference between crops reflects their biological differences, growing requirements, and market focus.
            
            {highest_range_category} demonstrates the greatest yield variability (range of {highest_range:.2f} tonnes/hectare), suggesting it may be more sensitive to growing conditions or management practices.
            
            Crops with higher median values relative to their means may indicate more consistent production regardless of external factors, making them potentially more reliable for farmers.
            """)
        else:  # Crop_Year
            # Check if there's a trend over years
            years = numeric_group_data['Year'].astype(int).tolist()
            means = numeric_group_data['Mean Yield'].tolist()
            
            # Simple trend detection
            if len(years) > 2:
                if means[-1] > means[0]:
                    trend = "increasing"
                elif means[-1] < means[0]:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                # Calculate average annual change
                total_change = means[-1] - means[0]
                years_diff = years[-1] - years[0]
                annual_change = total_change / years_diff if years_diff > 0 else 0
            else:
                trend = "undetermined"
                annual_change = 0
            
            st.info(f"""
            **📅 Yearly Yield Insight:**
            
            The data shows a generally {trend} trend in crop yields over the selected time period, with an average annual change of {annual_change:.3f} tonnes/hectare.
            
            The highest average yield was recorded in {max_mean_category} ({max_mean_value:.2f} tonnes/hectare), while the lowest was in {min_mean_category} ({min_mean_value:.2f} tonnes/hectare).
            
            Year {highest_range_category} shows the greatest yield variability (range of {highest_range:.2f} tonnes/hectare), which might indicate unusual weather conditions or policy changes affecting agriculture that year.
            
            Years with smaller differences between mean and median yields typically represent more normal growing conditions, while larger differences may indicate years with localized crop failures or exceptional harvests.
            """)
        
        # Add specific insight for crop comparison if filtering by crop
        if group_by == 'Crop' and selected_crop != 'All':
            st.info(f"""
            💡 **Focused Insight:** You're currently viewing only {selected_crop} data. To compare different crops, change the Crop filter to 'All' in the sidebar.
            """)

# Graphical Analysis Tab
with tab_graphical:
    st.header("Graphical Analysis")
    
    # Create subtabs for different visualization types
    viz_tabs = st.tabs(["Bar Charts", "Pie Charts", "Distributions"])
    
    # Helper function to calculate confidence intervals
    def calculate_ci(data, confidence=0.95):
        """Calculate confidence interval for a data series"""
        n = len(data)
        m = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return m, m - h, m + h  # mean, lower bound, upper bound
    
    # Bar Charts Tab
    with viz_tabs[0]:
        st.subheader("Bar Chart Analysis")
        
        # Create options for bar chart types
        bar_type = st.radio(
            "Select Bar Chart Type",
            ["Simple Bar Chart", "Multiple Bar Chart", "Component/Stacked Bar Chart"],
            horizontal=True
        )
        
        # Option to display confidence intervals
        show_ci = st.checkbox("Show Confidence Intervals", value=True)
        
        # Confidence level selection
        if show_ci:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[0.80, 0.85, 0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda x: f"{int(x*100)}%"
            )
        
        if bar_type == "Simple Bar Chart":
            st.write("#### Simple Bar Chart")
            
            # Select category for x-axis
            cat_var = st.selectbox(
                "Select Category Variable (X-axis)",
                ["District", "Crop", "Crop_Year"],
                key="simple_bar_x"
            )
            
            # Select numeric variable for y-axis
            y_metric = st.selectbox(
                "Select Metric (Y-axis)",
                ["Average Yield", "Count", "Average Area", "Average Production"],
                key="simple_bar_y"
            )
            
            # Calculate metrics based on selection with confidence intervals
            if y_metric == "Average Yield":
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['Yield']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average Yield": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['Yield'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average Yield"]
            elif y_metric == "Count":
                bar_data = filtered_df.groupby(cat_var).size().reset_index(name="Count")
                # No CI for counts
            elif y_metric == "Average Area":
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['Area']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average Area": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['Area'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average Area"]
            else:  # Average Production
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['Production']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average Production": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['Production'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average Production"]
                
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            if show_ci and y_metric != "Count":
                # Plot with error bars for confidence intervals
                y_col = bar_data.columns[1]
                
                # Sort data for better visualization if it's not crop year
                if cat_var != 'Crop_Year':
                    bar_data = bar_data.sort_values(by=y_col, ascending=False)
                
                # Limit number of districts shown if there are too many
                if cat_var == 'District' and len(bar_data) > 10:
                    bar_data = bar_data.head(10)
                    plt.title(f"{y_metric} by Top 10 Districts with {int(confidence_level*100)}% Confidence Intervals")
                else:
                    plt.title(f"{y_metric} by {cat_var} with {int(confidence_level*100)}% Confidence Intervals")
                
                # Calculate error bars AFTER sorting and limiting the data
                yerr = [bar_data[y_col] - bar_data["CI Lower"], bar_data["CI Upper"] - bar_data[y_col]]
                plt.bar(bar_data[cat_var], bar_data[y_col], yerr=yerr, capsize=10)
            else:
                # Regular bar chart without CI
                # Sort data for better visualization if it's not crop year
                if cat_var != 'Crop_Year' and y_metric != "Count":
                    bar_data = bar_data.sort_values(by=bar_data.columns[1], ascending=False)
                
                # Limit number of districts shown if there are too many
                if cat_var == 'District' and len(bar_data) > 10:
                    bar_data = bar_data.head(10)
                    plt.title(f"{y_metric} by Top 10 Districts")
                else:
                    plt.title(f"{y_metric} by {cat_var}")
                
                sns.barplot(data=bar_data, x=cat_var, y=bar_data.columns[1], ax=ax)
                
            plt.xlabel(cat_var)
            plt.ylabel(y_metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show data table below the chart
            st.write("Data used for chart:")
            st.dataframe(bar_data, use_container_width=True)
            
            # Add explanation of confidence intervals if they're being shown
            if show_ci and y_metric != "Count":
                st.info(f"""
                **Understanding Confidence Intervals:**
                
                The error bars represent the {int(confidence_level*100)}% confidence interval for each group's mean value.
                This means we are {int(confidence_level*100)}% confident that the true population mean falls within this range.
                
                Wider intervals indicate less certainty in the estimate, typically due to higher variance or smaller sample sizes.
                When confidence intervals for two groups don't overlap, it suggests a statistically significant difference between them.
                """)
            
        elif bar_type == "Multiple Bar Chart":
            st.write("#### Multiple Bar Chart (Grouped)")
            
            # Select category for x-axis
            primary_cat = st.selectbox(
                "Select Primary Category (X-axis)",
                ["District", "Crop_Year"],
                key="multi_bar_x"
            )
            
            # Select category for grouping
            secondary_cat = st.selectbox(
                "Select Secondary Category (Groups)",
                ["Crop"],
                key="multi_bar_group"
            )
            
            # Select metric for y-axis
            y_metric = st.selectbox(
                "Select Metric (Y-axis)",
                ["Average Yield", "Count", "Average Area", "Average Production"],
                key="multi_bar_y"
            )
            
            # Calculate data for the grouped bar chart with confidence intervals
            if y_metric == "Average Yield":
                if show_ci and y_metric != "Count":
                    # Get all combinations of primary and secondary categories
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    # Limit to top districts if needed
                    if primary_cat == 'District' and len(primary_cats) > 5:
                        # Find top districts by yield
                        top_districts = filtered_df.groupby('District')['Yield'].mean().nlargest(5).index.tolist()
                        primary_cats = top_districts
                    
                    # Calculate mean and CI for each combination
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['Yield']
                            
                            if len(subset) > 1:  # Need at least 2 points for CI
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Yield": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower  # For error bars
                                })
                            elif len(subset) == 1:  # Only one data point, no CI
                                mean = subset.iloc(0)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Yield": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0  # No error for single point
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    # Limit to top districts if needed
                    if primary_cat == 'District':
                        # Find top districts by yield
                        top_districts = filtered_df.groupby('District')['Yield'].mean().nlargest(5).index.tolist()
                        temp_df = filtered_df[filtered_df['District'].isin(top_districts)]
                        multi_bar_data = temp_df.groupby([primary_cat, secondary_cat])['Yield'].mean().reset_index()
                    else:
                        multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['Yield'].mean().reset_index()
                    
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average Yield"]
            elif y_metric == "Count":
                # Limit to top districts if needed
                if primary_cat == 'District':
                    # Find top districts by yield
                    top_districts = filtered_df.groupby('District').size().nlargest(5).index.tolist()
                    temp_df = filtered_df[filtered_df['District'].isin(top_districts)]
                    multi_bar_data = temp_df.groupby([primary_cat, secondary_cat]).size().reset_index(name="Count")
                else:
                    multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat]).size().reset_index(name="Count")
            elif y_metric == "Average Area":
                if show_ci and y_metric != "Count":
                    # Similar approach as for yield
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    # Limit to top districts if needed
                    if primary_cat == 'District' and len(primary_cats) > 5:
                        top_districts = filtered_df.groupby('District')['Area'].mean().nlargest(5).index.tolist()
                        primary_cats = top_districts
                    
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['Area']
                            
                            if len(subset) > 1:
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Area": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower
                                })
                            elif len(subset) == 1:
                                mean = subset.iloc(0)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Area": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    # Limit to top districts if needed
                    if primary_cat == 'District':
                        top_districts = filtered_df.groupby('District')['Area'].mean().nlargest(5).index.tolist()
                        temp_df = filtered_df[filtered_df['District'].isin(top_districts)]
                        multi_bar_data = temp_df.groupby([primary_cat, secondary_cat])['Area'].mean().reset_index()
                    else:
                        multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['Area'].mean().reset_index()
                    
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average Area"]
            else:  # Average Production
                if show_ci and y_metric != "Count":
                    # Similar approach as for yield
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    # Limit to top districts if needed
                    if primary_cat == 'District' and len(primary_cats) > 5:
                        top_districts = filtered_df.groupby('District')['Production'].mean().nlargest(5).index.tolist()
                        primary_cats = top_districts
                    
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['Production']
                            
                            if len(subset) > 1:
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Production": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower
                                })
                            elif len(subset) == 1:
                                mean = subset.iloc(0)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Production": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    # Limit to top districts if needed
                    if primary_cat == 'District':
                        top_districts = filtered_df.groupby('District')['Production'].mean().nlargest(5).index.tolist()
                        temp_df = filtered_df[filtered_df['District'].isin(top_districts)]
                        multi_bar_data = temp_df.groupby([primary_cat, secondary_cat])['Production'].mean().reset_index()
                    else:
                        multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['Production'].mean().reset_index()
                    
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average Production"]
                
            # Create multi-bar chart
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            if show_ci and y_metric != "Count":
                # Create multi-bar chart with error bars
                data_col = multi_bar_data.columns[2]  # Either "Average Yield", "Average Area", or "Average Production"
                
                # Get all categories
                all_primary_cats = sorted(multi_bar_data[primary_cat].unique())
                all_secondary_cats = sorted(multi_bar_data[secondary_cat].unique())
                
                # Set width and positions
                width = 0.8 / len(all_secondary_cats)
                x = np.arange(len(all_primary_cats))
                
                # Plot each secondary category
                for i, s_cat in enumerate(all_secondary_cats):
                    cat_data = multi_bar_data[multi_bar_data[secondary_cat] == s_cat]
                    cat_data = cat_data.set_index(primary_cat).reindex(all_primary_cats).reset_index()
                    cat_data = cat_data.fillna(0)  # Fill missing combinations with 0
                    
                    # Calculate error bars AFTER reindexing and filling
                    yerr = [(cat_data[data_col] - cat_data["CI Lower"]).values, 
                           (cat_data["CI Upper"] - cat_data[data_col]).values]
                    
                    positions = x + (i - len(all_secondary_cats)/2 + 0.5) * width
                    ax.bar(positions, cat_data[data_col], width=width, label=s_cat,
                          yerr=yerr, capsize=5)
                
                ax.set_xticks(x)
                ax.set_xticklabels(all_primary_cats)
                
                if primary_cat == 'District':
                    plt.title(f"{y_metric} by Top Districts and {secondary_cat} with {int(confidence_level*100)}% CI")
                else:
                    plt.title(f"{y_metric} by {primary_cat} and {secondary_cat} with {int(confidence_level*100)}% CI")
            else:
                # Regular multi-bar chart without CI
                sns.barplot(data=multi_bar_data, x=primary_cat, y=multi_bar_data.columns[2], hue=secondary_cat, ax=ax)
                
                if primary_cat == 'District':
                    plt.title(f"{y_metric} by Top Districts and {secondary_cat}")
                else:
                    plt.title(f"{y_metric} by {primary_cat} and {secondary_cat}")
                
            plt.xlabel(primary_cat)
            plt.ylabel(y_metric)
            plt.xticks(rotation=45)
            plt.legend(title=secondary_cat, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show data table below the chart
            st.write("Data used for chart:")
            st.dataframe(multi_bar_data, use_container_width=True)
            
            # Add explanation of confidence intervals if they're being shown
            if show_ci and y_metric != "Count":
                st.info(f"""
                **Understanding Confidence Intervals:**
                
                The error bars represent the {int(confidence_level*100)}% confidence interval for each group's mean value.
                This means we are {int(confidence_level*100)}% confident that the true population mean falls within this range.
                
                When confidence intervals for two groups don't overlap, it suggests a statistically significant difference between them.
                Wider intervals typically indicate smaller sample sizes or higher variability within the group.
                """)
        else:  # Component/Stacked Bar Chart
            st.write("#### Component/Stacked Bar Chart")
            
            # Select category for x-axis
            primary_cat = st.selectbox(
                "Select Primary Category (X-axis)",
                ["District", "Crop_Year"],
                key="stack_bar_x"
            )
            
            # Select category for stacking
            secondary_cat = st.selectbox(
                "Select Secondary Category (Stacks)",
                ["Crop"],
                key="stack_bar_stack"
            )
            
            # Limit to top districts if needed
            if primary_cat == 'District':
                top_districts = filtered_df.groupby('Area').sum().nlargest(10).index.tolist()
                stack_df = filtered_df[filtered_df['District'].isin(top_districts)]
            else:
                stack_df = filtered_df
                
            # Calculate percentage data
            # Get total area for each combination
            area_sums = pd.pivot_table(
                stack_df, 
                values='Area', 
                index=primary_cat,
                columns=secondary_cat, 
                aggfunc='sum'
            ).fillna(0)
            
            # Convert to percentages
            percentages = area_sums.div(area_sums.sum(axis=1), axis=0) * 100
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(8, 4.5))
            percentages.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            
            if primary_cat == 'District':
                plt.title(f"Proportion of Area by Crop for Top Districts")
            else:
                plt.title(f"Proportion of Area by {secondary_cat} for each {primary_cat}")
                
            plt.xlabel(primary_cat)
            plt.ylabel("Percentage of Total Area (%)")
            plt.xticks(rotation=45)
            plt.legend(title=secondary_cat, fontsize='small')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show raw counts and percentages
            col1, col2 = st.columns(2)
            with col1:
                st.write("Raw Area Values (hectares):")
                st.dataframe(area_sums, use_container_width=True)
            with col2:
                st.write("Percentages (%):")
                st.dataframe(percentages.round(1), use_container_width=True)
    
    # Pie Charts Tab
    with viz_tabs[1]:
        st.subheader("Pie Chart Analysis")
        
        # Add metric selection for pie chart
        pie_metric = st.selectbox(
            "Select Metric for Pie Chart",
            ["Area", "Production", "Yield", "Count"],
            key="pie_metric"
        )
        
        # Select category for pie chart
        pie_var = st.selectbox(
            "Select Category Variable",
            ["District", "Crop", "Year_groups"],
            key="pie_var"
        )
        
        if pie_var == "Year_groups":
            # Create year groups if not already present
            year_bins = list(range(1995, 2025, 5))
            year_labels = [f"{y}-{y+4}" for y in year_bins[:-1]]
            filtered_df['Year_group'] = pd.cut(filtered_df['Crop_Year'], bins=year_bins, labels=year_labels, right=False)
            
            # Calculate total by year group based on selected metric
            if pie_metric == "Count":
                pie_data = filtered_df.groupby('Year_group').size()
            else:
                pie_data = filtered_df.groupby('Year_group')[pie_metric].sum()
        elif pie_var == "District":
            # Limit to top districts for clarity based on selected metric
            if pie_metric == "Count":
                top_districts = filtered_df.groupby('District').size().nlargest(8).index
                pie_data_full = filtered_df.groupby('District').size()
            else:
                top_districts = filtered_df.groupby('District')[pie_metric].sum().nlargest(8).index
                pie_data_full = filtered_df.groupby('District')[pie_metric].sum()
            
            # Create 'Others' category for remaining districts
            pie_data = pd.Series({
                **{district: pie_data_full[district] for district in top_districts},
                'Others': pie_data_full[~pie_data_full.index.isin(top_districts)].sum()
            })
        else:
            # Calculate based on selected metric
            if pie_metric == "Count":
                pie_data = filtered_df.groupby(pie_var).size()
            else:
                pie_data = filtered_df.groupby(pie_var)[pie_metric].sum()
            
        # Create pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 9})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        metric_label = "Count" if pie_metric == "Count" else pie_metric
        
        if pie_var == "District":
            plt.title(f"Distribution of {metric_label} by Top Districts")
        else:
            plt.title(f"Distribution of {metric_label} by {pie_var.replace('_', ' ').title()}")
            
        st.pyplot(fig)
        
        # Show data table below the chart
        st.write("Data used for chart:")
        pie_data_df = pie_data.reset_index()
        pie_data_df.columns = [pie_var if pie_var != "Year_groups" else "Year Period", f"Total {metric_label}"]
        pie_data_df["Percentage"] = (pie_data_df[f"Total {metric_label}"] / pie_data_df[f"Total {metric_label}"].sum() * 100).round(2).astype(str) + '%'
        st.dataframe(pie_data_df, use_container_width=True)
        
    # Distributions Tab
    with viz_tabs[2]:
        st.subheader("Distribution Analysis")
        
        # Select chart type
        dist_chart_type = st.radio(
            "Select Distribution Chart Type",
            ["Histogram", "Box Plot"],
            horizontal=True
        )
        
        if dist_chart_type == "Histogram":
            st.write("#### Histogram")
            
            # Select numerical variable
            num_var = st.selectbox(
                "Select Numerical Variable",
                ["Yield", "Area", "Production", "Crop_Year"],
                key="hist_var"
            )
            
            # Optional: Segmentation
            use_segment = st.checkbox("Segment by Category", value=False)
            
            if use_segment:
                segment_var = st.selectbox(
                    "Select Segmentation Variable",
                    ["District", "Crop"],
                    key="hist_segment"
                )
                
                # For district segmentation, limit to top districts
                if segment_var == "District":
                    top_districts = filtered_df.groupby('District')[num_var].mean().nlargest(5).index.tolist()
                    hist_df = filtered_df[filtered_df['District'].isin(top_districts)]
                    
                    # Create segmented histogram with KDE
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    sns.histplot(data=hist_df, x=num_var, hue=segment_var, kde=True, multiple="stack", ax=ax)
                    plt.title(f"Distribution of {num_var} by Top Districts")
                else:
                    # Create segmented histogram with KDE
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    sns.histplot(data=filtered_df, x=num_var, hue=segment_var, kde=True, multiple="stack", ax=ax)
                    plt.title(f"Distribution of {num_var} by {segment_var}")
                
                plt.xlabel(num_var)
                plt.ylabel("Frequency")
                plt.legend(fontsize='small')
                plt.tight_layout()
            else:
                # Create simple histogram with KDE
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.histplot(data=filtered_df, x=num_var, kde=True, ax=ax)
                plt.title(f"Distribution of {num_var}")
                plt.xlabel(num_var)
                plt.ylabel("Frequency")
                plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display summary statistics
            st.write("Summary Statistics:")
            st.dataframe(filtered_df[num_var].describe().to_frame().T, use_container_width=True)
            
        else:  # Box Plot
            st.write("#### Box Plot")
            
            # Select numerical variable for box plot
            num_var = st.selectbox(
                "Select Numerical Variable",
                ["Yield", "Area", "Production"],
                key="box_var"
            )
            
            # Select categorical variable for grouping
            cat_var = st.selectbox(
                "Group by Category",
                ["Crop", "District"],
                key="box_cat"
            )
            
            # For district grouping, limit to top districts
            if cat_var == "District":
                top_districts = filtered_df.groupby('District')[num_var].median().nlargest(10).index.tolist()
                box_df = filtered_df[filtered_df['District'].isin(top_districts)]
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=box_df, x=cat_var, y=num_var, ax=ax)
                plt.title(f"Box Plot of {num_var} by Top 10 Districts")
            else:
                # Create box plot
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=filtered_df, x=cat_var, y=num_var, ax=ax)
                plt.title(f"Box Plot of {num_var} by {cat_var}")
            
            plt.xlabel(cat_var)
            plt.ylabel(num_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display summary statistics by group
            st.write("Summary Statistics by Group:")
            if cat_var == "District":
                summary = box_df.groupby(cat_var)[num_var].describe()
            else:
                summary = filtered_df.groupby(cat_var)[num_var].describe()
            st.dataframe(summary, use_container_width=True)

# After the existing Graphical Analysis Tab code, add the new Probability Methods Tab
with tab_probability:
    st.header("Normal Distribution Analysis")
    st.write("Analyze and fit Normal probability distribution to the crop data.")

    # Distribution type selection
    dist_section, param_section = st.columns([2, 1])
    
    with dist_section:
        # Select variable to analyze
        prob_var = st.selectbox(
            "Select Variable for Distribution Analysis",
            ["Yield", "Area", "Production"],
            key="prob_var"
        )
        
        # Optional filter by crop and district
        col1, col2 = st.columns(2)
        with col1:
            plot_by = st.radio("Plot distribution by:", ["All Data", "Crop", "District"])
        
        with col2:
            if plot_by == "Crop":
                selected_item = st.selectbox("Select Crop:", sorted(filtered_df['Crop'].unique()))
                plot_data = filtered_df[filtered_df['Crop'] == selected_item][prob_var].dropna()
                title_suffix = f"for {selected_item}"
            elif plot_by == "District":
                districts = sorted(filtered_df['District'].unique())
                if len(districts) > 10:
                    top_districts = filtered_df.groupby('District')[prob_var].mean().nlargest(10).index.tolist()
                    selected_item = st.selectbox("Select District:", top_districts)
                else:
                    selected_item = st.selectbox("Select District:", districts)
                plot_data = filtered_df[filtered_df['District'] == selected_item][prob_var].dropna()
                title_suffix = f"for {selected_item} district"
            else:
                plot_data = filtered_df[prob_var].dropna()
                title_suffix = "for all data"
    
    with param_section:
        st.subheader("Normal Distribution Parameters")
        
        # Fit distribution and show parameters
        if len(plot_data) > 0:
            # Fit normal distribution
            mu, sigma = stats.norm.fit(plot_data)
            st.metric("Mean (μ)", f"{mu:.4f}")
            st.metric("Std Dev (σ)", f"{sigma:.4f}")
            dist = stats.norm(mu, sigma)
            param_text = f"μ = {mu:.4f}, σ = {sigma:.4f}"
            
            # Goodness of fit test
            ks_statistic, p_value = stats.kstest(plot_data, dist.cdf)
            st.metric("K-S Test p-value", f"{p_value:.4f}")
            if p_value < 0.05:
                st.info("The data does not follow the Normal distribution (p < 0.05)")
            else:
                st.success("The data likely follows the Normal distribution (p >= 0.05)")
    
    # Create distribution plots
    if len(plot_data) > 0:
        # Plot histogram with fitted distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        sns.histplot(plot_data, kde=False, stat="density", alpha=0.6, ax=ax)
        
        # Generate points for distribution curve
        x = np.linspace(plot_data.min(), plot_data.max(), 1000)
        y = dist.pdf(x)
        
        # Plot the PDF
        plt.plot(x, y, 'r-', lw=2, label=f'Fitted Normal PDF\n{param_text}')
        plt.legend()
        plt.title(f'Normal Distribution Fit for {prob_var} {title_suffix}')
        plt.xlabel(prob_var)
        plt.ylabel('Density')
        plt.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Probability calculations
        st.subheader("Probability Calculations")
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability of being less than X
            less_than_value = st.number_input(
                f"Probability of {prob_var} being less than:",
                min_value=float(plot_data.min()),
                max_value=float(plot_data.max()),
                value=float(plot_data.median()),
                step=0.1
            )
            prob_less = dist.cdf(less_than_value)
            st.metric(f"P({prob_var} < {less_than_value:.2f})", f"{prob_less:.4f}")
        
        with col2:
            # Probability of being greater than X
            greater_than_value = st.number_input(
                f"Probability of {prob_var} being greater than:",
                min_value=float(plot_data.min()),
                max_value=float(plot_data.max()),
                value=float(plot_data.median()),
                step=0.1,
                key="greater_than"
            )
            prob_greater = 1 - dist.cdf(greater_than_value)
            st.metric(f"P({prob_var} > {greater_than_value:.2f})", f"{prob_greater:.4f}")
        
        # Quantile (Percentile) calculation
        st.subheader("Percentile Calculator")
        percentile = st.slider(
            "Select percentile:",
            min_value=1,
            max_value=99,
            value=50,
            step=1
        )
        
        quantile_value = dist.ppf(percentile/100)
        st.metric(f"{percentile}th Percentile of {prob_var}", f"{quantile_value:.4f}")
        
        # Add CDF plot
        st.subheader("Cumulative Distribution Function")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate ECDF
        sorted_data = np.sort(plot_data)
        ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        
        # Plot ECDF
        plt.step(sorted_data, ecdf, label='Empirical CDF', where='post')
        
        # Plot theoretical CDF
        x = np.linspace(plot_data.min(), plot_data.max(), 1000)
        plt.plot(x, dist.cdf(x), 'r-', lw=2, label='Theoretical Normal CDF')
        
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(f'CDF for {prob_var} {title_suffix}')
        plt.xlabel(prob_var)
        plt.ylabel('Cumulative Probability')
        st.pyplot(fig)

# Regression Modeling and Predictions Tab
with tab_regression:
    st.header("Regression Modeling and Predictions")
    st.write("Build and evaluate regression models to analyze relationships and make predictions.")
    
    # Linear Regression Analysis
    st.subheader("Linear Regression Analysis")
    
    # Model configuration section
    st.write("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target variable (Y)
        target_var = st.selectbox(
            "Select Target Variable (Y)",
            ["Yield", "Production"],
            key="target_var"
        )
        
        # Select predictor variable (X)
        predictor_options = ["Area", "Crop_Year"]
        if target_var == "Production":
            predictor_options.append("Yield")
        
        predictor_var = st.selectbox(
            "Select Predictor Variable (X)",
            predictor_options,
            key="predictor_var"
        )
        
        # Filter selection
        reg_filter = st.selectbox(
            "Filter Data By",
            ["None", "Crop", "District"],
            key="reg_filter"
        )
        
        if reg_filter == "Crop":
            reg_filter_value = st.selectbox(
                "Select Crop",
                sorted(filtered_df['Crop'].unique()),
                key="reg_filter_value"
            )
            reg_data = filtered_df[filtered_df['Crop'] == reg_filter_value]
            title_suffix = f"for {reg_filter_value}"
        elif reg_filter == "District":
            reg_filter_value = st.selectbox(
                "Select District",
                sorted(filtered_df['District'].unique()),
                key="reg_filter_value"
            )
            reg_data = filtered_df[filtered_df['District'] == reg_filter_value]
            title_suffix = f"for {reg_filter_value} district"
        else:
            reg_data = filtered_df.copy()
            title_suffix = "for all data"
    
    with col2:
        # Model type
        model_type = st.radio(
            "Regression Type",
            ["Simple Linear", "Polynomial"],
            key="model_type"
        )
        
        if model_type == "Polynomial":
            poly_degree = st.slider(
                "Polynomial Degree",
                min_value=2,
                max_value=5,
                value=2,
                key="poly_degree"
            )
        
        # Train-test split option
        use_train_test = st.checkbox(
            "Use Train-Test Split",
            value=True,
            key="use_train_test"
        )
        
        if use_train_test:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                key="test_size"
            ) / 100
    
    # Check if we have enough data
    if len(reg_data) < 10:
        st.warning("Not enough data for regression analysis with current filters.")
    else:
        # Prepare data
        X = reg_data[predictor_var].values.reshape(-1, 1)
        y = reg_data[target_var].values
        
        # Create and fit the model
        if model_type == "Simple Linear":
            if use_train_test:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            else:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
        else:  # Polynomial
            if use_train_test:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                poly = PolynomialFeatures(degree=poly_degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred_train = model.predict(X_train_poly)
                y_pred_test = model.predict(X_test_poly)
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            else:
                poly = PolynomialFeatures(degree=poly_degree)
                X_poly = poly.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Display model results
        st.write("### Model Results")
        
        # Display coefficients and equation
        if model_type == "Simple Linear":
            st.write(f"**Model Equation:** {target_var} = {model.intercept_:.4f} + {model.coef_[0]:,.4f} × {predictor_var}")
        else:
            equation = f"{target_var} = {model.intercept_:.4f}"
            for i, coef in enumerate(model.coef_[1:]):
                if i == 0:
                    equation += f" + {coef:.4f} × {predictor_var}"
                else:
                    equation += f" + {coef:.4f} × {predictor_var}^{i+1}"
            st.write(f"**Model Equation:** {equation}")
        
        # Display metrics
        metric_cols = st.columns(2 if use_train_test else 2)
        
        if use_train_test:
            metric_cols[0].metric("R² (Training)", f"{r2_train:.4f}")
            metric_cols[1].metric("R² (Test)", f"{r2_test:.4f}")
            metric_cols[0].metric("RMSE (Training)", f"{rmse_train:.4f}")
            metric_cols[1].metric("RMSE (Test)", f"{rmse_test:.4f}")
        else:
            metric_cols[0].metric("R² (All Data)", f"{r2:.4f}")
            metric_cols[1].metric("RMSE (All Data)", f"{rmse:.4f}")
        
        # Interpretation of R²
        r2_value = r2_test if use_train_test else r2
        if r2_value >= 0.75:
            r2_interpretation = "strong"
        elif r2_value >= 0.5:
            r2_interpretation = "moderate"
        elif r2_value >= 0.25:
            r2_interpretation = "weak"
        else:
            r2_interpretation = "very weak"
        
        st.info(f"""
        **Model Interpretation:**
        
        The R² value of {r2_value:.4f} indicates a {r2_interpretation} relationship between {predictor_var} and {target_var}.
        This means that approximately {r2_value*100:.1f}% of the variance in {target_var} can be explained by {predictor_var}.
        
        {'The model performs similarly on training and test data, suggesting good generalization.' if use_train_test and abs(r2_train - r2_test) < 0.1 else 
         'The difference between training and test performance suggests some overfitting.' if use_train_test else ''}
        """)
        
        # Plot the regression
        st.write("### Regression Plot")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(X, y, alpha=0.5, color='blue', label='Actual data')
        
        # Line for predicted values
        if model_type == "Simple Linear":
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_range = model.predict(X_range)
            plt.plot(X_range, y_range, color='red', linewidth=2, label='Regression line')
        else:
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            X_range_poly = poly.transform(X_range)
            y_range = model.predict(X_range_poly)
            plt.plot(X_range, y_range, color='red', linewidth=2, label=f'Polynomial (degree {poly_degree})')
        
        plt.xlabel(predictor_var)
        plt.ylabel(target_var)
        plt.title(f"{model_type} Regression of {target_var} vs {predictor_var} {title_suffix}")
        plt.grid(alpha=0.3)
        plt.legend()
        st.pyplot(fig)
        
        # Residual plot
        st.write("### Residual Analysis")
        
        if use_train_test:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Training residuals
            if model_type == "Simple Linear":
                residuals_train = y_train - model.predict(X_train)
            else:
                residuals_train = y_train - model.predict(X_train_poly)
            
            ax1.scatter(y_pred_train, residuals_train, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='-')
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Training Set Residuals")
            ax1.grid(alpha=0.3)
            
            # Test residuals
            if model_type == "Simple Linear":
                residuals_test = y_test - model.predict(X_test)
            else:
                residuals_test = y_test - model.predict(X_test_poly)
            
            ax2.scatter(y_pred_test, residuals_test, alpha=0.5, color='orange')
            ax2.axhline(y=0, color='r', linestyle='-')
            ax2.set_xlabel("Predicted Values")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Test Set Residuals")
            ax2.grid(alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if model_type == "Simple Linear":
                residuals = y - model.predict(X)
            else:
                residuals = y - model.predict(X_poly)
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")
            plt.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Residual distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if use_train_test:
            residuals = np.concatenate([residuals_train, residuals_test])
        
        sns.histplot(residuals, kde=True, ax=ax)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Residuals")
        plt.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Prediction for new values
        st.write("### Make Predictions")
        
        new_x = st.number_input(
            f"Enter a new {predictor_var} value:",
            min_value=float(X.min()),
            max_value=float(X.max()) * 1.5,
            value=float(X.mean()),
            step=0.1
        )
        
        # Make prediction
        if model_type == "Simple Linear":
            prediction = model.predict(np.array([[new_x]]))[0]
        else:
            prediction = model.predict(poly.transform(np.array([[new_x]])))[0]
        
        st.metric(f"Predicted {target_var} for {predictor_var} = {new_x}", f"{prediction:.4f}")


