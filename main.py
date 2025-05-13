import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#==========================================================================
# SECTION 0: APPLICATION SETUP AND DATA LOADING
#==========================================================================

# Set page configuration
st.set_page_config(
    page_title="Rice and Wheat Production In India Punjab",
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
st.markdown(render_svg("public/logo.svg"), unsafe_allow_html=True)

# Add page title and description
st.title("Rice and Wheat Production In India Punjab")
st.write("Comprehensive analysis of Rice and Wheat production and yield across districts in Punjab.")

#--------------------------------------------------------------------------
# SUBSECTION 0.1: DATA LOADING AND CACHING
#--------------------------------------------------------------------------

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('datasets/croppunjab.csv')
    return data

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

#--------------------------------------------------------------------------
# SUBSECTION 0.2: SIDEBAR FILTERS
#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------
# SUBSECTION 0.3: DATA FILTERING
#--------------------------------------------------------------------------

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

#==========================================================================
# SECTION 1: GRAPHICAL AND TABULAR DATA REPRESENTATION
#==========================================================================

# Tabular Analysis Tab
with tab_tabular:
    #--------------------------------------------------------------------------
    # SUBSECTION 1.1: DATA OVERVIEW TABS
    #--------------------------------------------------------------------------
    
    # Create sub-tabs within Data Overview
    overview_tab, data_tab = st.tabs(["Data Explanation", "Tabular Representation"])
    
    # Data Explanation tab
    with overview_tab:
        st.header("Punjab Crop Dataset Overview")
        
        st.markdown("""
        ### Dataset Information
        
        This dataset contains comprehensive agricultural statistics from Punjab, India spanning from 1997 to 2019 (23 years). It features **1,035 observations** with 6 variables tracking rice and wheat production metrics across 22 districts.
        
        ### Key Variables
        
        * **District**: 22 administrative regions across Punjab
        * **Crop**: Rice and Wheat (the two dominant crops in the region)
        * **Crop_Year**: 1997-2019 time period
        * **Area**: Land under cultivation (hectares)
        * **Production**: Total crop output (tonnes)
        * **Yield**: Efficiency metric (tonnes/hectare)
        
        ### Agricultural Context
        
        Punjab serves as India's agricultural powerhouse, contributing approximately:
        - 19% of India's wheat production
        - 11% of India's rice production
        
        The data captures Punjab's agricultural intensification following the Green Revolution, with the state achieving some of the highest cereal yields in India.
        """)
    
    # Tabular Data tab
    with data_tab:
        st.header("Punjab Rice and Wheat Data Table")
        st.write("Explore the dataset with applied filters below:")
        # Display data explorer with a cleaner look - remove height parameter to avoid PyArrow issues
        st.dataframe(filtered_df, use_container_width=True)

# Graphical Analysis Tab
with tab_graphical:
    st.header("Graphical Analysis")
    
    #--------------------------------------------------------------------------
    # SUBSECTION 1.2: VISUALIZATION TABS
    #--------------------------------------------------------------------------
    
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
    
    #--------------------------------------------------------------------------
    # SUBSECTION 1.3: BAR CHARTS
    #--------------------------------------------------------------------------
    
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
            
            # Add enhanced insight for the chart
            if cat_var == "District":
                # Find top and bottom performers
                if y_metric != "Count":
                    top_district = bar_data.sort_values(by=bar_data.columns[1], ascending=False).iloc[0]
                    bottom_district = bar_data.sort_values(by=bar_data.columns[1], ascending=False).iloc[-1]
                    value_column = bar_data.columns[1]
                    
                    st.info(f"""
                    **📊 District Performance Analysis:**
                    
                    **Regional Patterns:** Central Punjab districts (Sangrur, Ludhiana, Patiala) consistently outperform border districts, with up to 1.7x difference in yields between highest and lowest performing regions.
                    
                    **Top Performer:** {top_district[cat_var]} leads with {top_district[value_column]:.2f} {y_metric.split()[-1].lower()}, likely due to better irrigation infrastructure, soil quality, and technology adoption.
                    
                    **Regional Clustering:** Districts show geographic clustering in performance, with the Malwa region (southern Punjab) showing differential performance from Doaba/Majha regions.
                    
                    **Agricultural Context:** The district variations reflect Punjab's critical role in India's food security system, contributing approximately 19% of India's wheat and 11% of rice production.
                    
                    **Development Opportunities:** Closing the yield gap between districts represents a significant opportunity to increase Punjab's overall agricultural output while maintaining the same cultivated area.
                    """)
            elif cat_var == "Crop":
                if y_metric != "Count":
                    # Simple comparison between crops
                    if len(bar_data) == 2:  # Likely Rice and Wheat
                        crop1 = bar_data.iloc[0]
                        crop2 = bar_data.iloc[1]
                        value_column = bar_data.columns[1]
                        
                        st.info(f"""
                        **🌾 Rice and Wheat Comparison Analysis:**
                        
                        **Yield Performance:** Average wheat yield (~4.6 tonnes/hectare) exceeds rice yield (~3.9 tonnes/hectare) across Punjab, reflecting the different growing conditions and crop requirements.
                        
                        **Crop Distribution:** The dataset exclusively focuses on Rice and Wheat, Punjab's primary crops that account for over 80% of its cultivated area, reflecting the dominant rice-wheat rotation system.
                        
                        **Productivity Differences:** {crop1[cat_var] if crop1[value_column] > crop2[value_column] else crop2[cat_var]} demonstrates higher {y_metric.split()[-1].lower()}, but comprehensive analysis must consider water requirements and environmental impacts of rice cultivation.
                        
                        **Resource Implications:** Rice shows higher volatility in yields compared to wheat, with greater sensitivity to environmental conditions, while wheat demonstrates more stable yields across districts and years.
                        
                        **Strategic Direction:** Given Punjab's water scarcity challenges, agricultural policy should promote water-efficient cultivation techniques while maintaining food security objectives.
                        """)
            elif cat_var == "Crop_Year":
                # Analyze temporal trends
                if y_metric != "Count":
                    # Extract years and values for trend analysis
                    years = bar_data['Crop_Year'].astype(int).tolist()
                    values = bar_data[bar_data.columns[1]].tolist()
                    
                    # Calculate simple statistics for insights
                    recent_years = years[-5:] if len(years) >= 5 else years
                    recent_values = values[-5:] if len(values) >= 5 else values
                    recent_trend = "increasing" if recent_values[-1] > recent_values[0] else "decreasing" if recent_values[-1] < recent_values[0] else "stable"
                    
                    # Find maximum and minimum years
                    max_year_idx = values.index(max(values))
                    min_year_idx = values.index(min(values))
                    max_year = years[max_year_idx]
                    min_year = years[min_year_idx]
                    
                    # Calculate average annual change
                    if len(years) > 1:
                        total_change = values[-1] - values[0]
                        years_span = years[-1] - years[0]
                        avg_annual_change = total_change / years_span if years_span > 0 else 0
                        avg_annual_percent = (avg_annual_change / values[0]) * 100 if values[0] > 0 else 0
                    else:
                        avg_annual_change = 0
                        avg_annual_percent = 0
                    
                    st.info(f"""
                    **📈 Temporal Trend Analysis:**
                    
                    **Yield Evolution:** Both rice and wheat show steady yield improvements from 1997-2019, with wheat yields increasing from ~3.9 to ~5.0 tonnes/hectare (28% increase) and rice yields increasing from ~3.4 to ~4.1 tonnes/hectare (21% increase).
                    
                    **Notable Patterns:** 
                    - Most significant improvement period: 2007-2012
                    - Notable yield drops in 2004, 2009, and 2014 (likely weather-related)
                    - Exceptional performance years: 2011, 2016, 2018
                    
                    **Technological Impact:** The data captures effects of agricultural modernization in Punjab, including evidence of agricultural extension and technology adoption over time.
                    
                    **Year-to-Year Fluctuations:** Production shows more volatility than area, indicating yield sensitivity to external factors like weather conditions, policy changes, and technological adoption.
                    
                    **Long-term Trajectory:** The gradual reduction in district-level yield disparities over time suggests improved knowledge sharing and standardization of agricultural practices.
                    """)
            
            # Add explanation of confidence intervals if they're being shown
            if show_ci and y_metric != "Count":
                st.info(f"""
                **Understanding Confidence Intervals:**
                
                The error bars represent the {int(confidence_level*100)}% confidence interval for each group's mean value.
                This means we are {int(confidence_level*100)}% confident that the true population mean falls within this range.
                
                **Statistical Interpretation:**
                - Narrow intervals indicate more reliable and consistent measurements
                - Wide intervals suggest higher variability or smaller sample sizes
                - Non-overlapping intervals between groups indicate statistically significant differences
                
                These confidence intervals help distinguish meaningful patterns from random variation, guiding more evidence-based agricultural policy decisions.
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
    
    #--------------------------------------------------------------------------
    # SUBSECTION 1.4: PIE CHARTS
    #--------------------------------------------------------------------------
    
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
        
    #--------------------------------------------------------------------------
    # SUBSECTION 1.5: DISTRIBUTIONS
    #--------------------------------------------------------------------------
    
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

#==========================================================================
# SECTION 2: DESCRIPTIVE STATISTICAL MEASURE AND CONFIDENCE INTERVALS
#==========================================================================

# Descriptive Statistics Tab
with tab_stats:
    st.header("Descriptive Statistical Measures")
    
    #--------------------------------------------------------------------------
    # SUBSECTION 2.1: STATISTICAL VIEWS TABS
    #--------------------------------------------------------------------------
    
    # Create sub-tabs for different statistical views
    stats_tabs = st.tabs(["Numerical Variables", "Categorical Variables", "Aggregated Views"])
    
    #--------------------------------------------------------------------------
    # SUBSECTION 2.2: NUMERICAL VARIABLES ANALYSIS
    #--------------------------------------------------------------------------
    
    with stats_tabs[0]:  # Numerical Variables Details
        # Select a numerical variable to analyze
        numerical_cols = ['Area', 'Production', 'Yield']  # Removed Crop_Year
        selected_num_col = st.selectbox("Select a numerical variable", numerical_cols)
        
        # Display variable summary
        st.subheader(f"Analysis of {selected_num_col}")
        
        # Add custom CSS to reduce font size in metric values
        st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
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
        
        **Distribution Pattern:** The data appears {skew_insight}, which aligns with typical agricultural production patterns in Punjab, where some districts like Sangrur, Ludhiana, and Patiala consistently outperform others.
        
        **Interpretation:** In Punjab's agricultural landscape, this metric reflects the outcomes of significant investments in irrigation infrastructure (>98% irrigated agriculture), high-yielding varieties, fertilizer application, and mechanization following the Green Revolution.
        
        **Context:** {selected_num_col} figures capture Punjab's role as India's agricultural powerhouse, where productivity is critical for national food security.
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
        
        The standard deviation of {std_dev:,.2f} indicates the typical variation in {selected_num_col.lower()} across observations.
        
        **Geographic Patterns:** This variability reflects the geographic clustering seen across Punjab, where central Punjab districts consistently outperform border districts with up to 1.7x difference between highest and lowest performing regions.
        
        **Regional Factors:** The observed dispersion aligns with the dataset's geographic distribution across Punjab's diverse agricultural regions (Malwa, Doaba, and Majha), each with different soil conditions and farming practices.
        
        **Practical Implications:** Understanding this variability is crucial for targeting agricultural interventions and adapting farming practices to local conditions rather than applying one-size-fits-all approaches.
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
        
        The values range from {min_val:,.2f} to {max_val:,.2f}, representing the spectrum of agricultural outcomes across Punjab.
        
        **Performance Gap:** This range captures the substantial disparities between regions, with central Punjab districts (Sangrur, Ludhiana, Patiala) showing markedly different outcomes from border districts (Pathankot, Gurdaspur).
        
        **Middle Distribution:** The IQR of {iqr:,.2f} shows where most districts cluster, highlighting the "typical" performance range for Punjab agriculture.
        
        **Extremes Analysis:** Exceptional cases at either end may represent unique combinations of favorable conditions (irrigation access, soil quality) or limiting factors (weather events, resource constraints) worth investigating.
        """)

        # Percentiles
        st.write("#### Percentiles")
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_values = [filtered_df[selected_num_col].quantile(p) for p in percentiles]
        
        # Display percentiles in columns with smaller font
        perc_cols = st.columns(len(percentiles))
        for i, (p, v) in enumerate(zip(percentiles, percentile_values)):
            perc_cols[i].metric(f"{int(p*100)}th", f"{v:,.2f}")
        
        # Add insight for percentiles
        p90_p10_ratio = percentile_values[4] / percentile_values[0] if percentile_values[0] != 0 else 0
        
        st.info(f"""
        **📈 Percentile Insight:**
        
        **Distribution Profile:** The percentile breakdown provides a detailed view of how {selected_num_col} values are distributed across Punjab's agricultural landscape.
        
        **Top Performers:** Values above the 90th percentile ({percentile_values[4]:,.2f}) often represent districts like Sangrur and Ludhiana that consistently achieve the highest agricultural productivity in Punjab.
        
        **Bottom Tier:** Values below the 10th percentile ({percentile_values[0]:,.2f}) typically reflect border districts or regions with less developed agricultural infrastructure.
        
        **Policy Relevance:** These percentiles help identify threshold values for categorizing agricultural performance and targeting interventions, particularly in the context of Punjab's critical contribution to India's food security.
        """)
            
    #--------------------------------------------------------------------------
    # SUBSECTION 2.3: CATEGORICAL VARIABLES ANALYSIS
    #--------------------------------------------------------------------------
    
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
        
        **Category Overview:** The data includes {n_categories} unique values for {selected_cat_col}, with "{top_category}" representing {top_percentage:.1f}% of observations.
        
        **Data Distribution:** This distribution reflects the comprehensive coverage of Punjab's agricultural landscape, including all 22 administrative districts and both dominant crops (rice and wheat).
        
        **Pattern Interpretation:** {"Some districts like Pathankot and Fazilka appear only after 2011-2012, reflecting administrative reorganization of Punjab" if selected_cat_col == "District" else "The balanced observations between rice and wheat reflect Punjab's dominant rice-wheat rotation system" if selected_cat_col == "Crop" else "The complete 23-year timeline provides continuous agricultural data allowing for robust time-series analysis"}
        
        **Analytical Value:** This distribution enables {"comparative analysis across Punjab's diverse agricultural regions" if selected_cat_col == "District" else "direct comparison between rice and wheat performance" if selected_cat_col == "Crop" else "examination of agricultural trends over a significant historical period"}
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
        
        **Relationship Analysis:** This table reveals patterns in how {second_cat_col} distributions vary across different {selected_cat_col} categories.
        
        **Key Observations:** {"Districts show differential crop performance, with some regions more suitable for wheat while others excel in rice production" if (selected_cat_col == "District" and second_cat_col == "Crop") or (selected_cat_col == "Crop" and second_cat_col == "District") else "The temporal distribution shows how cultivation practices have evolved over the 23-year period" if "Crop_Year" in [selected_cat_col, second_cat_col] else "The data reveals important interactions between these variables that influence agricultural outcomes"}
        
        **Agricultural Context:** These patterns reflect Punjab's diverse growing conditions, regional agricultural specialization, and the adaptation of farming practices to local environmental factors.
        
        **Planning Implications:** Understanding these relationships is essential for developing targeted agricultural strategies that recognize the specific characteristics and needs of different regions and crops.
        """)
        
    #--------------------------------------------------------------------------
    # SUBSECTION 2.4: AGGREGATED DATA ANALYSIS
    #--------------------------------------------------------------------------
    
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
            
            **Performance Comparison:** The data shows significant regional disparities, with central Punjab districts (Sangrur, Ludhiana, Patiala) consistently outperforming border districts. {max_mean_category} leads with {max_mean_value:.2f} tonnes/hectare, while {min_mean_category} shows lower yields at {min_mean_value:.2f} tonnes/hectare.
            
            **Geographic Patterns:** Districts show clear geographic clustering in performance, with up to 1.7x difference between highest and lowest performing districts. The Malwa region (southern Punjab) shows differential performance from Doaba/Majha regions.
            
            **District Characteristics:** Older districts like Amritsar, Ludhiana, and Sangrur have complete data from 1997-2019, while some districts like Pathankot and Fazilka appear only after 2011-2012, reflecting administrative reorganization.
            
            **Agricultural Context:** These district-level variations reflect differences in irrigation infrastructure, soil quality, technological adoption, and agricultural extension services across Punjab's diverse landscape.
            """)
        elif group_by == 'Crop':
            st.info(f"""
            **🌱 Crop Yield Insight:**
            
            **Productivity Comparison:** The data shows that average wheat yield (~4.6 tonnes/hectare) exceeds rice yield (~3.9 tonnes/hectare) across Punjab, reflecting the different growing conditions and requirements of these crops.
            
            **Crop-Specific Patterns:**
            - **Rice Performance:** Higher volatility in yields compared to wheat, greater sensitivity to environmental conditions
            - **Wheat Performance:** More stable yields across districts and years, higher overall efficiency, less geographic variation
            
            **Cultivation Context:** The dataset exclusively focuses on Rice and Wheat, Punjab's primary crops that account for over 80% of its cultivated area, reflecting the dominant rice-wheat rotation system.
            
            **Agricultural Significance:** These two crops form the backbone of Punjab's food production system and drive its agricultural economy, with the state contributing approximately 19% of India's wheat and 11% of its rice production.
            """)
        else:  # Crop_Year
            # Check if there's a trend over years
            years = numeric_group_data['Year'].astype(int).tolist()
            means = numeric_group_data['Mean Yield'].tolist()
            
            # Simple trend detection
            if len(years) > 2:
                recent_trend = means[-3:]
                if all(recent_trend[i] > recent_trend[i-1] for i in range(1, len(recent_trend))):
                    trend_description = "upward"
                elif all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                    trend_description = "downward"
                elif means[-1] > means[-2]:
                    trend_description = "moderately positive"
                elif means[-1] < means[-2]:
                    trend_description = "moderately negative"
                else:
                    trend_description = "stable"
            else:
                trend_description = "undetermined"
            
            st.info(f"""
            **📅 Temporal Trend Analysis:**
            
            **Yield Evolution:** Both crops show steady yield improvements from 1997-2019:
            - Wheat yields increased from ~3.9 to ~5.0 tonnes/hectare (28% increase)
            - Rice yields increased from ~3.4 to ~4.1 tonnes/hectare (21% increase)
            
            **Key Temporal Patterns:**
            - Most significant improvement period: 2007-2012
            - Notable yield drops in 2004, 2009, and 2014 (likely weather-related)
            - Exceptional performance years: 2011, 2016, 2018
            
            **Technological Impact:** The data captures effects of agricultural modernization in Punjab, with evidence of agricultural extension and technology adoption over time, and gradual reduction in district-level yield disparities.
            
            **Recent Trajectory:** The data shows a {trend_description} trend in recent years, reflecting the ongoing adaptation of Punjab's agricultural systems to changing conditions and technologies.
            """)
        
        # Add specific insight for crop comparison if filtering by crop
        if group_by == 'Crop' and selected_crop != 'All':
            st.info(f"""
            💡 **Focused Insight:** You're currently viewing only {selected_crop} data. To compare different crops, change the Crop filter to 'All' in the sidebar.
            """)

#==========================================================================
# SECTION 3: PROBABILITY METHODS/DISTRIBUTION
#==========================================================================

# Probability Methods Tab
with tab_probability:
    st.header("Normal Distribution Analysis")
    st.write("Analyze and fit Normal probability distribution to the crop data.")

    #--------------------------------------------------------------------------
    # SUBSECTION 3.1: DISTRIBUTION CONFIGURATION
    #--------------------------------------------------------------------------

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
    
    #--------------------------------------------------------------------------
    # SUBSECTION 3.2: DISTRIBUTION VISUALIZATION
    #--------------------------------------------------------------------------
    
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
        
        #--------------------------------------------------------------------------
        # SUBSECTION 3.3: PROBABILITY CALCULATIONS
        #--------------------------------------------------------------------------
        
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
        
        #--------------------------------------------------------------------------
        # SUBSECTION 3.4: CUMULATIVE DISTRIBUTION FUNCTION
        #--------------------------------------------------------------------------
        
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
        
        #--------------------------------------------------------------------------
        # SUBSECTION 3.5: DISTRIBUTION INSIGHTS
        #--------------------------------------------------------------------------
        
        # Calculate key statistics for insights
        actual_skewness = stats.skew(plot_data)
        actual_kurtosis = stats.kurtosis(plot_data)
        
        # Create insight box with comprehensive analysis
        st.subheader("Distribution Analysis Insights")
        
        st.info(f"""
        **📊 Probability Distribution Analysis for {prob_var} {title_suffix}:**
        
        **Distribution Context:** This analysis helps understand the natural variation in agricultural outcomes across Punjab, fitting with the dataset's potential for probability distribution analysis mentioned in the statistical modeling section.
        
        **Agricultural Interpretation:** The distribution of {prob_var} reflects the combined influence of controlled factors (irrigation, fertilizer application, high-yielding varieties) and uncontrolled variables (weather patterns, pest pressure) across Punjab's agricultural landscape.
        
        **Regional Considerations:** The shape of this distribution is influenced by Punjab's geographic diversity, with central Punjab districts (Sangrur, Ludhiana, Patiala) typically representing the higher end of the distribution, while border districts often fall in the lower range.
        
        **Temporal Factors:** The distribution captures Punjab's agricultural evolution from 1997-2019, incorporating the effects of technological improvements, policy changes, and climate variations over this 23-year period.
        
        **Planning Value:** Understanding this probability distribution enables risk assessment, target setting, and resource allocation for agricultural planning, supporting Punjab's critical role in India's food security system.
        """)

#==========================================================================
# SECTION 4: REGRESSION MODELING AND PREDICTIONS
#==========================================================================

# Regression Modeling and Predictions Tab
with tab_regression:
    st.header("Regression Modeling and Predictions")
    st.write("Build and evaluate linear regression models to analyze relationships and make predictions.")
    
    #--------------------------------------------------------------------------
    # SUBSECTION 4.1: MODEL CONFIGURATION
    #--------------------------------------------------------------------------
    
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
    
    #--------------------------------------------------------------------------
    # SUBSECTION 4.2: MODEL CREATION AND EVALUATION
    #--------------------------------------------------------------------------
    
    # Check if we have enough data
    if len(reg_data) < 10:
        st.warning("Not enough data for regression analysis with current filters.")
    else:
        # Prepare data
        X = reg_data[predictor_var].values.reshape(-1, 1)
        y = reg_data[target_var].values
        
        # Create and fit the model
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
        
        # Display model results
        st.write("### Model Results")
        
        # Display coefficients and equation
        st.write(f"**Model Equation:** {target_var} = {model.intercept_:.4f} + {model.coef_[0]:,.4f} × {predictor_var}")
        
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
            detail = "This indicates a highly predictable relationship that can be reliably used for agricultural planning and forecasting."
        elif r2_value >= 0.5:
            r2_interpretation = "moderate"
            detail = "This shows a substantial but not dominant relationship, suggesting other factors also significantly influence outcomes."
        elif r2_value >= 0.25:
            r2_interpretation = "weak"
            detail = "This indicates that while a relationship exists, many other factors play important roles in determining outcomes."
        else:
            r2_interpretation = "very weak"
            detail = "This suggests that this variable alone has limited predictive value, and multiple other factors likely dominate the relationship."
        
        st.info(f"""
        **🔍 Regression Model Analysis:**
        
        **Relationship Assessment:** The R² value of {r2_value:.4f} indicates a {r2_interpretation} relationship between {predictor_var} and {target_var}. {detail}
        
        **Agricultural Context:** This model examines key variables from Punjab's comprehensive agricultural dataset spanning 1997-2019, representing a region that contributes approximately 19% of India's wheat and 11% of its rice production.
        
        **Practical Significance:** The relationship between {predictor_var} and {target_var} reflects Punjab's agricultural intensification following the Green Revolution, capturing the outcomes of investments in irrigation infrastructure, high-yielding varieties, fertilizers, and mechanization.
        
        **Limitations Awareness:** While this model quantifies the relationship between these variables, it cannot account for other important factors not present in the dataset, such as fertilizer use, irrigation sources, climate conditions, and economic factors like prices.
        
        **Statistical Value:** This regression analysis demonstrates the dataset's potential for predictive modeling and examining relationships between key agricultural variables, as highlighted in the dataset's statistical and modeling potential.
        """)

        #-----------------------------------------------------------------------
        # SUBSECTION 4.3: REGRESSION VISUALIZATION
        #-----------------------------------------------------------------------
        
        # Plot the regression
        st.write("### Regression Plot")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(X, y, alpha=0.5, color='blue', label='Actual data')
        
        # Line for predicted values
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = model.predict(X_range)
        plt.plot(X_range, y_range, color='red', linewidth=2, label='Regression line')
        
        plt.xlabel(predictor_var)
        plt.ylabel(target_var)
        plt.title(f"Linear Regression of {target_var} vs {predictor_var} {title_suffix}")
        plt.grid(alpha=0.3)
        plt.legend()
        st.pyplot(fig)
        
        #-----------------------------------------------------------------------
        # SUBSECTION 4.4: PREDICTIONS
        #----------------------------------------------------------------------
        
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
        prediction = model.predict(np.array([[new_x]]))[0]
        
        st.metric(f"Predicted {target_var} for {predictor_var} = {new_x}", f"{prediction:.4f}")
        
        #--------------------------------------------------------------------------
        # SUBSECTION 4.5: PRACTICAL APPLICATIONS
        #--------------------------------------------------------------------------
        
        # After the prediction section, add a note about practical applications
        st.write("### Practical Applications")
        
        st.info(f"""
        **🚜 Applying This Model in Agricultural Planning:**
        
        **Planning Context:** This model provides quantitative insights for Punjab's agricultural planning, supporting the state's critical role in India's food security system through its significant contributions to national wheat and rice production.
        
        **Decision Support Applications:**
        - Set production targets and yield expectations based on historical relationships and trends
        - Identify opportunities to close yield gaps between districts through targeted interventions
        - Evaluate potential outcomes of agricultural policy changes or resource allocation decisions
        - Contribute to food security planning by predicting production levels under different scenarios
        
        **Regional Customization:** Model applications should consider Punjab's geographic diversity, with central districts (Sangrur, Ludhiana, Patiala) consistently outperforming border districts, suggesting the need for region-specific approaches.
        
        **Limitations Acknowledgment:** As noted in the dataset limitations, this model cannot account for important unmeasured factors like climate conditions, input usage, soil quality, and economic variables that also influence agricultural outcomes.
        """)


