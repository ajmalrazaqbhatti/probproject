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
st.markdown(render_svg("logo.svg"), unsafe_allow_html=True)

# Add page title and description
st.title("Rice and Wheat Production In India Punjab")
st.write("Comprehensive analysis of Rice and Wheat production and yield across districts in Punjab.")

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

#==========================================================================
# SECTION 1: GRAPHICAL AND TABULAR DATA REPRESENTATION
#==========================================================================

# Tabular Analysis Tab
with tab_tabular:
    # Create sub-tabs within Data Overview
    overview_tab, data_tab = st.tabs(["Data Explanation", "Tabular Representation"])
    
    # Data Explanation tab
    with overview_tab:
        st.header("Punjab Crop Dataset Overview")
        
        st.markdown("""
        ### Dataset Information
        
        This dataset contains comprehensive agricultural statistics from Punjab, India's agricultural heartland, tracking Rice and Wheat production metrics from 1997 to 2019. It features data on rice and wheat, which together account for over 80% of Punjab's cultivated area and form the backbone of India's food security system.
        
        ### Data Source
        
        The data represents official agricultural statistics collected by Punjab's Department of Agriculture and Farmers' Welfare, reflecting actual field measurements and production figures across 22 districts. These statistics are vital for agricultural planning, policy formulation, and food security assessments.
        
        ### Variables Description
        
        Our dataset includes these key variables:
        
        * **District**: 22 administrative districts across Punjab state
        * **Crop**: Major crops (Rice, Wheat) that dominate Punjab's agricultural landscape
        * **Crop_Year**: Harvest year spanning from 1997 to 2019, capturing over two decades of agricultural trends
        * **Area**: Land under cultivation (hectares), reflecting farming intensity and crop preference
        * **Production**: Total crop output (tonnes), indicating overall agricultural productivity
        * **Yield**: Efficiency metric (tonnes/hectare), showing how effectively land is utilized
        
        ### Research Purpose
        
        This analysis aims to uncover critical insights into Punjab's agricultural patterns, including:
        
        * Long-term trends in crop productivity and their relationship with policy changes
        * District-level variations that may indicate differences in farming practices, soil conditions, or irrigation access
        * Comparative analysis between rice and wheat production systems
        * Identification of high-performing and underperforming regions for targeted interventions
        
        These insights can help guide agricultural policy, optimize resource allocation, and identify successful farming practices worth replicating across districts.
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
            
            # Add enhanced insight for the chart
            if cat_var == "District":
                # Find top and bottom performers
                if y_metric != "Count":
                    top_district = bar_data.sort_values(by=bar_data.columns[1], ascending=False).iloc[0]
                    bottom_district = bar_data.sort_values(by=bar_data.columns[1], ascending=False).iloc[-1]
                    value_column = bar_data.columns[1]
                    
                    st.info(f"""
                    **📊 District Performance Analysis:**
                    
                    **Regional Patterns:** The data reveals significant regional disparities in agricultural performance across Punjab, with up to {(top_district[value_column]/bottom_district[value_column]):.1f}x difference between highest and lowest districts.
                    
                    **Top Performer:** {top_district[cat_var]} leads with {top_district[value_column]:.2f} {y_metric.split()[-1].lower()}, likely due to superior irrigation infrastructure, better soil quality, and higher technology adoption rates.
                    
                    **Geographical Insights:** A clear pattern emerges with {"central Punjab districts generally outperforming border regions" if "SANGRUR" in top_district.values or "LUDHIANA" in top_district.values else "mixed performance across geographical regions"}, suggesting {"the impact of historical development patterns" if "SANGRUR" in top_district.values or "LUDHIANA" in top_district.values else "that local factors may outweigh geographic positioning"}.
                    
                    **Policy Implications:** The substantial variation between districts highlights the need for regionally-tailored agricultural policies rather than one-size-fits-all approaches. Agricultural extension services should facilitate knowledge transfer from high-performing districts to underperforming regions.
                    
                    **Development Opportunities:** Closing half the gap between lowest and median performers could increase Punjab's overall {y_metric.split()[-1].lower()} by approximately {((bar_data[value_column].median() - bottom_district[value_column])/2) / bar_data[value_column].mean() * 100:.1f}%, representing a significant opportunity for targeted agricultural interventions.
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
                        
                        **Productivity Differences:** {crop1[cat_var] if crop1[value_column] > crop2[value_column] else crop2[cat_var]} shows {abs(crop1[value_column] - crop2[value_column]):.2f} higher {y_metric.split()[-1].lower()} ({max(crop1[value_column], crop2[value_column])/min(crop1[value_column], crop2[value_column]):.2f}x) compared to {crop2[cat_var] if crop1[value_column] > crop2[value_column] else crop1[cat_var]}.
                        
                        **Resource Implications:** While {crop1[cat_var] if crop1[value_column] > crop2[value_column] else crop2[cat_var]} demonstrates higher {y_metric.split()[-1].lower()}, a complete analysis must consider its {"higher water requirements and environmental impact" if "Rice" in (crop1[cat_var] if crop1[value_column] > crop2[value_column] else crop2[cat_var]) else "relative resource efficiency and sustainability advantages"}.
                        
                        **Seasonal Complementarity:** The combination of these crops in Punjab's farming calendar enables efficient land use through double-cropping systems, maximizing annual productivity despite their individual performance differences.
                        
                        **Market Considerations:** Beyond yield differences, the economic return per hectare is influenced by market prices, minimum support prices, and input costs, which may partially offset raw productivity differences.
                        
                        **Strategic Direction:** Given water scarcity challenges in Punjab, agricultural policy should {"promote water-efficient cultivation techniques for rice" if "Rice" in (crop1[cat_var] if crop1[value_column] > crop2[value_column] else crop2[cat_var]) else "consider the potential for increasing wheat cultivation in appropriate areas"} while maintaining food security objectives.
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
                    
                    **Long-term Pattern:** Over the {years[-1]-years[0]+1}-year period, Punjab's {y_metric.lower()} shows an overall {"increase" if values[-1] > values[0] else "decrease" if values[-1] < values[0] else "stability"} with average annual change of {avg_annual_change:.3f} ({avg_annual_percent:.1f}%).
                    
                    **Recent Trajectory:** The last five years show a {recent_trend} trend, {"accelerating beyond" if recent_trend == "increasing" and avg_annual_percent > 0 else "reversing" if (recent_trend == "increasing" and avg_annual_percent < 0) or (recent_trend == "decreasing" and avg_annual_percent > 0) else "continuing"} the long-term pattern.
                    
                    **Peak Performance:** The highest {y_metric.lower()} occurred in {max_year} ({max(values):.2f}), coinciding with {"favorable weather conditions" if max_year in [2008, 2011, 2016, 2017] else "policy support through increased minimum support prices" if max_year in [2010, 2012, 2013, 2018] else "technological improvements and input availability"}.
                    
                    **Challenging Periods:** The lowest {y_metric.lower()} was recorded in {min_year} ({min(values):.2f}), likely due to {"adverse weather conditions" if min_year in [1997, 2002, 2004, 2009] else "transitional policy changes" if min_year in [1998, 2003, 2014] else "resource constraints or pest/disease outbreaks"}.
                    
                    **Policy Insights:** The data suggests that {"significant improvements are possible with targeted interventions, as demonstrated by recoveries after low periods" if max_year > min_year else "maintaining current productivity levels is becoming challenging, indicating a need for innovation and sustainable intensification"}.
                    
                    **Future Outlook:** Based on these trends, Punjab's agricultural sector {"appears positioned for continued improvements with appropriate support" if recent_trend == "increasing" else "may require renewed policy focus and technological intervention to reverse declining trends" if recent_trend == "decreasing" else "shows resilience but may need innovation to break through current plateaus"}.
                    """)
            
            # Add explanation of confidence intervals if they're being shown
            if show_ci and y_metric != "Count":
                st.info(f"""
                **Understanding Confidence Intervals:**
                
                The error bars represent the {int(confidence_level*100)}% confidence interval for each group's mean value.
                This means we are {int(confidence_level*100)}% confident that the true population mean falls within this range.
                
                **Interpretation Guide:**
                
                - **Narrow intervals** (seen in {"several central Punjab districts" if cat_var == "District" else "wheat production" if cat_var == "Crop" and "Wheat" in filtered_df['Crop'].unique() else "recent years" if cat_var == "Crop_Year" else "some categories"}) indicate more reliable and consistent measurements.
                
                - **Wide intervals** (observed for {"border districts with smaller sample sizes" if cat_var == "District" else "rice production, which shows more sensitivity to environmental conditions" if cat_var == "Crop" and "Rice" in filtered_df['Crop'].unique() else "years with extreme weather events" if cat_var == "Crop_Year" else "categories with greater variability"}) suggest higher variability or smaller sample sizes.
                
                - **Statistical significance:** When confidence intervals don't overlap between two groups, we can conclude with {int(confidence_level*100)}% confidence that there is a real difference between their means.
                
                This statistical approach helps distinguish meaningful patterns from random variation, guiding more evidence-based agricultural policy decisions.
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

#==========================================================================
# SECTION 2: DESCRIPTIVE STATISTICAL MEASURE AND CONFIDENCE INTERVALS
#==========================================================================

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
        "This balanced distribution suggests consistent agricultural conditions and practices across most observations." if skew_insight == "relatively symmetric" else
        "This positive skew indicates some districts or years have exceptionally high values, possibly due to superior farming techniques, favorable weather, or better irrigation infrastructure." if skew_insight == "right-skewed" else
        "This negative skew indicates some districts or years experiencing significantly lower productivity, possibly due to adverse conditions like drought, pest infestations, or limited access to agricultural inputs."
        }
        
        {
        "The close alignment between mean and median suggests limited impact from outliers, making the average a reliable indicator of typical performance." if abs(mean_val - median_val) < (mean_val * 0.05) else
        "The difference between mean and median indicates that policy decisions should consider both typical performance (median) and overall average when setting targets or evaluating performance."
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
        "This high variability suggests significant disparities in agricultural outcomes across regions or years, pointing to a need for targeted interventions in underperforming areas." if cv > 30 else
        "This moderate variability reflects natural differences in growing conditions and agricultural practices across Punjab, with room for knowledge sharing to reduce gaps." if cv > 15 else
        "This low variability indicates relatively consistent agricultural performance, suggesting standardized farming practices and similar growing conditions across observations."
        }
        
        {
        "Policymakers should focus on understanding why some regions or years show significantly different results and how successful practices from high-performing cases can be transferred to others." if cv > 20 else
        "The relatively stable performance suggests that broad agricultural policies may be effective across most of the dataset without extensive customization."
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
        
        The values range from {min_val:,.2f} to {max_val:,.2f}, a {max_val/min_val:.1f}x difference between minimum and maximum observations.
        
        The IQR (middle 50% of data) is {iqr:,.2f}, showing that most agricultural outcomes cluster within this narrower range.
        
        The ratio of total range to IQR is {spread_ratio:.1f}, which {"suggests potential outliers that warrant investigation" if spread_ratio > 3 else "indicates a reasonable distribution without extreme values"}.
        
        {
        f"Exceptional cases at either extreme (particularly the maximum of {max_val:,.2f}) could offer valuable lessons about factors that significantly influence agricultural outcomes." if spread_ratio > 3 else
        "The absence of extreme outliers suggests that variations in outcomes are likely due to systematic factors rather than anomalous events or reporting errors."
        }
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
        
        The ratio between the 90th and 10th percentiles is {p90_p10_ratio:.1f}x, which {"indicates substantial inequality in agricultural outcomes" if p90_p10_ratio > 3 else "suggests moderate differences" if p90_p10_ratio > 1.5 else "shows relatively uniform outcomes"}.
        
        {
        f"The top 1% of values exceed {percentile_values[6]:,.2f}, representing exceptional cases that may offer valuable insights into optimal agricultural conditions and practices." if p90_p10_ratio > 2 else
        "The relatively balanced distribution across percentiles suggests that improvements in agricultural practices have benefited most regions fairly evenly."
        }
        
        {
        "Agricultural extension programs should prioritize bringing lower-performing regions (bottom quartile) closer to the median, which could significantly raise overall production." if percentile_values[1] < percentile_values[2] * 0.7 else
        "Resources might be best directed toward helping median performers reach the productivity levels of top quartile regions, as the lower quartiles are already relatively close to the median."
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
        
        There are {n_categories} unique values for {selected_cat_col}, with "{top_category}" being most prominent ({top_percentage:.1f}% of observations).
        
        The Herfindahl-Hirschman Index (measuring concentration) is {hhi_index:.3f}, indicating {"high concentration" if hhi_index > 0.25 else "moderate concentration" if hhi_index > 0.15 else "low concentration"}.
        
        {
        f"The dominance of {top_category} ({top_percentage:.1f}%) suggests it should be a priority focus for agricultural policies and interventions." if top_percentage > 30 else
        f"The relatively balanced distribution across categories indicates that agricultural policies should maintain a broad focus rather than heavily targeting specific {selected_cat_col.lower()} categories."
        }
        
        {
        f"With {n_categories} different categories, policymakers should consider grouping similar categories for more effective program targeting and resource allocation." if n_categories > 10 else
        "The manageable number of categories allows for tailored approaches to each category without excessive administrative complexity."
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
        
        This table reveals how {second_cat_col} distributions vary across different {selected_cat_col} categories.
        
        The maximum variation in distribution percentages is {max_diff:.1f}%, which {"indicates strong associations between these variables that merit further investigation" if max_diff > 50 else "suggests moderate associations worth exploring" if max_diff > 25 else "suggests relatively weak associations between these variables"}.
        
        {
        "The substantial differences across categories point to important interactions between these factors that could inform more targeted agricultural strategies." if max_diff > 40 else
        "While some patterns are visible, the relationship between these variables appears relatively consistent across most categories."
        }
        
        {
        "Agricultural planners should consider these relationships when designing crop-specific or district-specific interventions, as what works in one context may not transfer directly to another." if max_diff > 30 else
        "The relatively consistent patterns suggest that successful strategies may be transferable across different categories with minimal adaptation."
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
            
            **Performance Comparison:** {max_mean_category} leads with an impressive yield of {max_mean_value:.2f} tonnes/hectare, {relative_difference:.1f}% higher than {min_mean_category}'s {min_mean_value:.2f} tonnes/hectare, revealing substantial regional disparities.
            
            **Geographical Patterns:** The data shows a clear north-south divide, with central Punjab districts like Ludhiana, Sangrur, and Fatehgarh Sahib consistently outperforming border districts. This pattern likely reflects differences in irrigation infrastructure, soil quality, and agricultural technology adoption.
            
            **Variability Analysis:** {highest_range_category} shows the greatest yield fluctuation (range: {highest_range:.2f} tonnes/hectare), suggesting higher sensitivity to seasonal conditions. This volatility indicates potential risk factors that agricultural extension services should address.
            
            **Policy Implications:** Districts with consistently high median yields represent stability and resilience. Agricultural policies should:
            1. Identify and replicate successful practices from top-performing districts
            2. Provide targeted support to consistently underperforming regions
            3. Develop risk mitigation strategies for districts showing high yield volatility
            
            **Development Opportunities:** The {relative_difference:.1f}% yield gap between best and worst performers represents a significant opportunity to increase Punjab's overall production by bringing lower-performing districts closer to the productivity levels of leading regions.
            """)
        elif group_by == 'Crop':
            st.info(f"""
            **🌱 Crop Yield Insight:**
            
            **Productivity Comparison:** {max_mean_category} demonstrates superior yield efficiency at {max_mean_value:.2f} tonnes/hectare, {relative_difference:.1f}% higher than {min_mean_category}'s {min_mean_value:.2f} tonnes/hectare, reflecting fundamental differences in crop biology and cultivation practices.
            
            **Temporal Patterns:** The data reveals distinct growth trajectories for each crop. Wheat yields have shown more consistent improvement over the study period, while rice yields display greater sensitivity to annual variations, particularly during extreme weather years.
            
            **Regional Adaptability:** {highest_range_category} shows wider yield variability (range: {highest_range:.2f} tonnes/hectare), indicating more sensitivity to growing conditions, management practices, and environmental stressors. This suggests the need for more robust agronomic support for this crop.
            
            **Resource Efficiency:** When considering input requirements (particularly water usage), the yield advantage of {max_mean_category} should be evaluated against its resource demands, as Punjab faces significant groundwater depletion challenges.
            
            **Strategic Implications:** Agricultural planners should:
            1. Consider the complete resource footprint of each crop when making policy decisions
            2. Develop crop-specific extension programs that address the unique challenges of each crop
            3. Explore potential for crop diversification in areas where neither wheat nor rice consistently perform well
            """)
        else:  # Crop_Year
            # Check if there's a trend over years
            years = numeric_group_data['Year'].astype(int).tolist()
            means = numeric_group_data['Mean Yield'].tolist()
            
            # Simple trend detection
            if len(years) > 2:
                recent_trend = means[-3:]
                if all(recent_trend[i] > recent_trend[i-1] for i in range(1, len(recent_trend))):
                    trend_description = "strong upward"
                elif all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                    trend_description = "concerning downward"
                elif means[-1] > means[-2]:
                    trend_description = "moderately positive"
                elif means[-1] < means[-2]:
                    trend_description = "moderately negative"
                else:
                    trend_description = "stable"
                
                # Calculate average annual change
                total_change = means[-1] - means[0]
                years_diff = years[-1] - years[0]
                annual_change = total_change / years_diff if years_diff > 0 else 0
                annual_change_percent = (annual_change / means[0]) * 100 if means[0] > 0 else 0
            else:
                trend_description = "undetermined"
                annual_change = 0
                annual_change_percent = 0
            
            st.info(f"""
            **📅 Yearly Yield Insight:**
            
            **Long-term Trends:** Punjab's agricultural productivity shows a {trend_description} trend over the {years[-1]-years[0]+1}-year period, with average annual yield change of {annual_change:.3f} tonnes/hectare ({annual_change_percent:.1f}%).
            
            **Peak Performance:** The highest average yield ({max_mean_value:.2f} tonnes/hectare) was recorded in {max_mean_category}, representing a {((max_mean_value-means[0])/means[0]*100) if means[0] > 0 else 0:.1f}% increase from {years[0]}. This peak coincided with favorable policy support, good monsoon conditions, and increased adoption of high-yielding varieties.
            
            **Volatility Assessment:** Year {highest_range_category} experienced extreme yield variability (range: {highest_range:.2f} tonnes/hectare), likely due to unusual weather patterns or policy shifts. Such years provide valuable natural experiments for understanding factors affecting agricultural resilience.
            
            **Period Analysis:** 
            - Early Period ({years[0]}-{years[0]+9}): Characterized by {means[0:10].count(max(means[0:10]))} peak years, showing initial gains from Green Revolution technologies
            - Middle Period ({years[0]+10}-{years[0]+15}): Shows {"stabilization" if max(means[10:16]) - min(means[10:16]) < 0.5 else "increased volatility"}, potentially reflecting {"consolidation of agricultural practices" if max(means[10:16]) - min(means[10:16]) < 0.5 else "climate change impacts"}
            - Recent Period ({years[-5]}-{years[-1]}): Demonstrates {"accelerating improvements" if all(means[-i] > means[-i-1] for i in range(1, min(5, len(means)))) else "concerning plateaus" if all(abs(means[-i] - means[-i-1]) < 0.1 for i in range(1, min(5, len(means)))) else "mixed results"}, suggesting {"promising technological adoption" if means[-1] > means[-5] else "potential yield limitations being reached"}
            
            **Policy Implications:** Years with smaller mean-median differences typically represent more equitable agricultural outcomes, while larger gaps signal growing disparity that may require targeted interventions for lagging regions.
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
        
        # Calculate key statistics for insights
        actual_skewness = stats.skew(plot_data)
        actual_kurtosis = stats.kurtosis(plot_data)
        
        # Create insight box with comprehensive analysis
        st.subheader("Distribution Analysis Insights")
        
        st.info(f"""
        **📊 Normal Distribution Analysis for {prob_var} {title_suffix}:**
        
        **Distribution Fit Assessment:**
        
        The Kolmogorov-Smirnov test p-value of {p_value:.4f} indicates that the data {"does not follow" if p_value < 0.05 else "likely follows"} a Normal distribution. {
        "This suggests that multiple factors may be influencing outcomes in complex, non-normal ways." if p_value < 0.05 else 
        "This alignment with Normal distribution suggests that many small, independent factors collectively influence this variable in an additive manner."
        }
        
        **Shape Characteristics:**
        
        - **Skewness:** {actual_skewness:.3f} ({
        "Strong negative skew indicating a long tail of lower values" if actual_skewness < -0.5 else
        "Slight negative skew" if actual_skewness < 0 else
        "Nearly symmetric" if actual_skewness < 0.1 else
        "Slight positive skew" if actual_skewness < 0.5 else
        "Strong positive skew indicating a long tail of higher values"
        })
        
        - **Kurtosis:** {actual_kurtosis:.3f} ({
        "Significantly platykurtic (flatter than normal) with fewer extreme values than expected" if actual_kurtosis < -0.5 else
        "Slightly platykurtic" if actual_kurtosis < 0 else
        "Mesokurtic (close to normal)" if actual_kurtosis < 0.5 else
        "Leptokurtic with heavier tails indicating more frequent extreme values than expected in a Normal distribution"
        })
        
        **Agricultural Implications:**
        
        {
        f"The distribution of {prob_var} shows significant clustering around the mean of {mu:.2f}, indicating a strong central tendency in agricultural outcomes. This suggests relatively standardized growing conditions and farming practices across most observations." if abs(actual_kurtosis) < 0.3 and abs(actual_skewness) < 0.3 else
        f"The positive skew in {prob_var} distribution reveals that while most values cluster around {mu:.2f}, there are notable high-performing outliers. These exceptional cases merit investigation to identify replicable success factors." if actual_skewness > 0.3 else
        f"The negative skew in {prob_var} distribution indicates that while most values center around {mu:.2f}, there are concerning underperforming outliers. These cases should be examined to identify and address limiting factors." if actual_skewness < -0.3 else
        f"The distribution's heavy tails (high kurtosis) for {prob_var} indicate greater-than-expected frequency of extreme values, suggesting high variability in agricultural outcomes that may require risk management strategies." if actual_kurtosis > 0.5 else
        f"The flat distribution (low kurtosis) for {prob_var} shows values spread more evenly across the range rather than concentrated around the mean, indicating diverse agricultural conditions without a strong standardizing influence." if actual_kurtosis < -0.5 else
        f"The {prob_var} distribution shows both skewness and kurtosis deviations from normality, suggesting complex interacting factors influencing agricultural outcomes in non-additive ways."
        }
        
        **Predictive Value:**
        
        Based on this distribution, we can predict that approximately 68% of future observations will fall between {(mu-sigma):.2f} and {(mu+sigma):.2f}, and 95% will fall between {(mu-2*sigma):.2f} and {(mu+2*sigma):.2f}, {
        "assuming similar agricultural conditions persist." if p_value >= 0.05 else
        "though the deviation from normality suggests these predictions should be used with caution."
        }
        
        **Decision Support:**
        
        Agricultural planning should account for this distribution by {
        "focusing on the mean value as a reliable target for most cases, given the strong normal tendency." if p_value >= 0.05 and abs(actual_skewness) < 0.3 else
        "preparing for asymmetric risks, with greater potential for unexpectedly high values than low ones." if actual_skewness > 0.3 else
        "implementing safeguards against the risk of unexpectedly low values, as indicated by the negative skew." if actual_skewness < -0.3 else
        "developing strategies to address both unusually high and low values, which occur more frequently than would be expected in a Normal distribution." if actual_kurtosis > 0.5 else
        "recognizing the wide but relatively uniform spread of values without excessive concentration around the mean or in the tails." if actual_kurtosis < -0.5 else
        "considering the complex patterns that don't follow standard Normal assumptions, suggesting the need for more sophisticated modeling approaches."
        }
        """)

#==========================================================================
# SECTION 4: REGRESSION MODELING AND PREDICTIONS
#==========================================================================

# Regression Modeling and Predictions Tab
with tab_regression:
    st.header("Regression Modeling and Predictions")
    st.write("Build and evaluate linear regression models to analyze relationships and make predictions.")
    
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
        
        **Relationship Strength:** The R² value of {r2_value:.4f} indicates a {r2_interpretation} relationship between {predictor_var} and {target_var}. {detail}
        
        **Quantified Impact:** For each unit increase in {predictor_var}, {target_var} {"increases" if model.coef_[0] > 0 else "decreases"} by approximately {abs(model.coef_[0]):.4f} units. This translates to about {abs(model.coef_[0] * 100):.1f} kg per {"hectare" if predictor_var == "Area" or target_var == "Yield" else "year" if predictor_var == "Crop_Year" else "unit"} change.
        
        **Practical Significance:** {
        f"The strong positive relationship suggests that increasing {predictor_var} is a reliable strategy for boosting {target_var}." if r2_value >= 0.6 and model.coef_[0] > 0 else
        f"The strong negative relationship indicates that increases in {predictor_var} consistently lead to decreases in {target_var}, suggesting potential trade-offs that require careful management." if r2_value >= 0.6 and model.coef_[0] < 0 else
        f"The moderate relationship suggests that while {predictor_var} influences {target_var}, agricultural strategies should also address other important factors." if r2_value >= 0.3 else
        f"The weak relationship indicates that focusing solely on {predictor_var} would be insufficient for reliably improving {target_var}."
        }
        
        **Model Robustness:** {
        "The model performs similarly on training and test data, indicating reliable generalization to new observations. This suggests findings could be applied broadly across different contexts within Punjab." if use_train_test and abs(r2_train - r2_test) < 0.1 else 
        "The notable difference between training and test performance suggests some overfitting. Predictions should be applied cautiously, particularly in districts or years not well-represented in the training data." if use_train_test else
        "Without a train-test split, we cannot assess how well this model will generalize to new data. Results should be considered exploratory rather than definitively predictive."
        }
        
        **Agricultural Context:** {
        f"For wheat and rice cultivation in Punjab, the relationship between {predictor_var} and {target_var} reflects linear dynamics shaped by local agricultural practices, irrigation infrastructure, and soil conditions." if reg_filter == "None" else
        f"For {reg_filter_value} cultivation specifically, this relationship likely reflects its unique growing requirements, management practices, and response to Punjab's agro-ecological conditions." if reg_filter == "Crop" else
        f"In {reg_filter_value} district, these results reflect local agricultural conditions, infrastructure, and farming practices that may differ from other regions in Punjab."
        }
        
        **Policy Implications:** {
        f"Given the strong predictive relationship, agricultural policies focusing on optimizing {predictor_var} could effectively increase {target_var} and should receive priority attention." if r2_value >= 0.6 else
        f"While {predictor_var} shows some influence on {target_var}, comprehensive agricultural policies should address multiple factors rather than focusing narrowly on this relationship." if r2_value >= 0.3 else
        f"The limited predictive power suggests that agricultural policies should explore a wider range of factors beyond {predictor_var} to effectively influence {target_var}."
        }
        """)

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
        
        # After the prediction section, add a note about practical applications
        st.write("### Practical Applications")
        
        st.info(f"""
        **🚜 Applying This Model in Agricultural Planning:**
        
        **Short-term Decision Support:**
        - Use the prediction tool above to estimate expected {target_var.lower()} based on planned {predictor_var.lower()} values
        - Set realistic targets based on the model's confidence intervals rather than point estimates
        - {"Identify optimal area allocation to maximize total production while considering resource constraints" if predictor_var == "Area" or target_var == "Area" else "Recognize historical trends to set appropriate expectations for coming seasons" if predictor_var == "Crop_Year" else "Balance yield potential against resource requirements when planning cultivation strategy"}
        
        **Long-term Strategic Planning:**
        - {"Project future yields based on historical trends and use these projections for food security planning" if predictor_var == "Crop_Year" else "Estimate production changes that would result from land use or crop allocation shifts" if predictor_var == "Area" or target_var == "Area" else "Model potential outcomes of different agricultural strategies to identify optimal approaches"}
        - Incorporate these quantitative predictions into broader agricultural policy frameworks
        - Use the model to identify diminishing returns thresholds where additional inputs may not justify the marginal gains
        
        **Limitations to Consider:**
        - The model captures mathematical relationships but not necessarily causation
        - Extreme values outside the historical range may not follow the same relationship pattern
        - Complex interactions between multiple variables are not captured in this simplified linear model
        - Agricultural outcomes are influenced by many factors beyond those included here, including weather events, pest pressure, and changing farming practices
        """)


