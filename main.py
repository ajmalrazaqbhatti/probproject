import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import scipy.stats as stats

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
                ["region", "sex", "smoker"],
                key="simple_bar_x"
            )
            
            # Select numeric variable for y-axis
            y_metric = st.selectbox(
                "Select Metric (Y-axis)",
                ["Average Charges", "Count", "Average BMI", "Average Age"],
                key="simple_bar_y"
            )
            
            # Calculate metrics based on selection with confidence intervals
            if y_metric == "Average Charges":
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['charges']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average Charges": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['charges'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average Charges"]
            elif y_metric == "Count":
                bar_data = filtered_df.groupby(cat_var).size().reset_index(name="Count")
                # No CI for counts
            elif y_metric == "Average BMI":
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['bmi']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average BMI": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['bmi'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average BMI"]
            else:  # Average Age
                if show_ci:
                    # Group data and calculate confidence intervals
                    ci_data = []
                    unique_categories = filtered_df[cat_var].unique()
                    
                    for category in unique_categories:
                        subset = filtered_df[filtered_df[cat_var] == category]['age']
                        mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                        ci_data.append({
                            cat_var: category,
                            "Average Age": mean,
                            "CI Lower": ci_lower,
                            "CI Upper": ci_upper
                        })
                    
                    bar_data = pd.DataFrame(ci_data)
                else:
                    bar_data = filtered_df.groupby(cat_var)['age'].mean().reset_index()
                    bar_data.columns = [cat_var, "Average Age"]
                
            # Create bar chart - Reduced size from (10, 6) to (8, 4.5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            if show_ci and y_metric != "Count":
                # Plot with error bars for confidence intervals
                y_col = bar_data.columns[1]
                yerr = [bar_data[y_col] - bar_data["CI Lower"], bar_data["CI Upper"] - bar_data[y_col]]
                plt.bar(bar_data[cat_var], bar_data[y_col], yerr=yerr, capsize=10)
                plt.title(f"{y_metric} by {cat_var.capitalize()} with {int(confidence_level*100)}% Confidence Intervals")
            else:
                # Regular bar chart without CI
                sns.barplot(data=bar_data, x=cat_var, y=bar_data.columns[1], ax=ax)
                plt.title(f"{y_metric} by {cat_var.capitalize()}")
                
            plt.xlabel(cat_var.capitalize())
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
                ["region", "sex", "smoker"],
                key="multi_bar_x"
            )
            
            # Select category for grouping
            secondary_cat = st.selectbox(
                "Select Secondary Category (Groups)",
                [cat for cat in ["region", "sex", "smoker"] if cat != primary_cat],
                key="multi_bar_group"
            )
            
            # Select metric for y-axis
            y_metric = st.selectbox(
                "Select Metric (Y-axis)",
                ["Average Charges", "Count", "Average BMI", "Average Age"],
                key="multi_bar_y"
            )
            
            # Calculate data for the grouped bar chart with confidence intervals
            if y_metric == "Average Charges":
                if show_ci and y_metric != "Count":
                    # Get all combinations of primary and secondary categories
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    # Calculate mean and CI for each combination
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['charges']
                            
                            if len(subset) > 1:  # Need at least 2 points for CI
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Charges": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower  # For error bars
                                })
                            elif len(subset) == 1:  # Only one data point, no CI
                                mean = subset.iloc[0]
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Charges": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0  # No error for single point
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['charges'].mean().reset_index()
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average Charges"]
            elif y_metric == "Count":
                multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat]).size().reset_index(name="Count")
                # No CI for counts
            elif y_metric == "Average BMI":
                if show_ci and y_metric != "Count":
                    # Similar approach as for charges
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['bmi']
                            
                            if len(subset) > 1:
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average BMI": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower
                                })
                            elif len(subset) == 1:
                                mean = subset.iloc[0]
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average BMI": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['bmi'].mean().reset_index()
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average BMI"]
            else:  # Average Age
                if show_ci and y_metric != "Count":
                    # Similar approach as for charges
                    primary_cats = filtered_df[primary_cat].unique()
                    secondary_cats = filtered_df[secondary_cat].unique()
                    
                    multi_ci_data = []
                    
                    for p_cat in primary_cats:
                        for s_cat in secondary_cats:
                            subset = filtered_df[(filtered_df[primary_cat] == p_cat) & 
                                              (filtered_df[secondary_cat] == s_cat)]['age']
                            
                            if len(subset) > 1:
                                mean, ci_lower, ci_upper = calculate_ci(subset, confidence=confidence_level)
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Age": mean,
                                    "CI Lower": ci_lower,
                                    "CI Upper": ci_upper,
                                    "CI Error": mean - ci_lower
                                })
                            elif len(subset) == 1:
                                mean = subset.iloc[0]
                                multi_ci_data.append({
                                    primary_cat: p_cat,
                                    secondary_cat: s_cat,
                                    "Average Age": mean,
                                    "CI Lower": mean,
                                    "CI Upper": mean,
                                    "CI Error": 0
                                })
                    
                    multi_bar_data = pd.DataFrame(multi_ci_data)
                else:
                    multi_bar_data = filtered_df.groupby([primary_cat, secondary_cat])['age'].mean().reset_index()
                    multi_bar_data.columns = [primary_cat, secondary_cat, "Average Age"]
                
            # Create multi-bar chart - Reduced size from (12, 7) to (8, 4.5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            if show_ci and y_metric != "Count":
                # Create multi-bar chart with error bars
                data_col = multi_bar_data.columns[2]  # Either "Average Charges", "Average BMI", or "Average Age"
                
                # Get all categories
                all_primary_cats = multi_bar_data[primary_cat].unique()
                all_secondary_cats = multi_bar_data[secondary_cat].unique()
                
                # Set width and positions
                width = 0.8 / len(all_secondary_cats)
                x = np.arange(len(all_primary_cats))
                
                # Plot each secondary category
                for i, s_cat in enumerate(all_secondary_cats):
                    cat_data = multi_bar_data[multi_bar_data[secondary_cat] == s_cat]
                    yerr = [(cat_data[data_col] - cat_data["CI Lower"]).values, 
                           (cat_data["CI Upper"] - cat_data[data_col]).values]
                    
                    positions = x + (i - len(all_secondary_cats)/2 + 0.5) * width
                    ax.bar(positions, cat_data[data_col], width=width, label=s_cat,
                          yerr=yerr, capsize=5)
                
                ax.set_xticks(x)
                ax.set_xticklabels(all_primary_cats)
                plt.title(f"{y_metric} by {primary_cat.capitalize()} and {secondary_cat.capitalize()} with {int(confidence_level*100)}% CI")
            else:
                # Regular multi-bar chart without CI
                sns.barplot(data=multi_bar_data, x=primary_cat, y=multi_bar_data.columns[2], hue=secondary_cat, ax=ax)
                plt.title(f"{y_metric} by {primary_cat.capitalize()} and {secondary_cat.capitalize()}")
                
            plt.xlabel(primary_cat.capitalize())
            plt.ylabel(y_metric)
            plt.xticks(rotation=45)
            plt.legend(title=secondary_cat.capitalize(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
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
                ["region", "sex"],
                key="stack_bar_x"
            )
            
            # Select category for stacking
            secondary_cat = st.selectbox(
                "Select Secondary Category (Stacks)",
                ["smoker"],
                key="stack_bar_stack"
            )
            
            # Calculate percentage data
            # Get counts for each combination
            counts = pd.crosstab(filtered_df[primary_cat], filtered_df[secondary_cat])
            
            # Convert to percentages
            percentages = counts.div(counts.sum(axis=1), axis=0) * 100
            
            # Create stacked bar chart - Reduced size from (10, 6) to (8, 4.5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            percentages.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            plt.title(f"Proportion of {secondary_cat.capitalize()} Status by {primary_cat.capitalize()}")
            plt.xlabel(primary_cat.capitalize())
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45)
            plt.legend(title=secondary_cat.capitalize(), fontsize='small')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show raw counts and percentages
            col1, col2 = st.columns(2)
            with col1:
                st.write("Raw counts:")
                st.dataframe(counts, use_container_width=True)
            with col2:
                st.write("Percentages (%):")
                st.dataframe(percentages.round(1), use_container_width=True)
    
    # Pie Charts Tab
    with viz_tabs[1]:
        st.subheader("Pie Chart Analysis")
        
        # Select category for pie chart
        pie_var = st.selectbox(
            "Select Category Variable",
            ["region", "sex", "smoker", "age_groups"],
            key="pie_var"
        )
        
        if pie_var == "age_groups":
            # Create age groups if not already present
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=age_bins, labels=age_labels, right=False)
            pie_data = filtered_df['age_group'].value_counts()
        else:
            pie_data = filtered_df[pie_var].value_counts()
            
        # Create pie chart - Reduced size from (8, 8) to (6, 6)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 9})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f"Distribution of {pie_var.replace('_', ' ').capitalize()}")
        st.pyplot(fig)
        
        # Show data table below the chart
        st.write("Data used for chart:")
        pie_data_df = pie_data.reset_index()
        pie_data_df.columns = [pie_var, "Count"]
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
                ["age", "bmi", "children", "charges"],
                key="hist_var"
            )
            
            # Optional: Segmentation
            use_segment = st.checkbox("Segment by Category", value=False)
            
            if use_segment:
                segment_var = st.selectbox(
                    "Select Segmentation Variable",
                    ["region", "sex", "smoker"],
                    key="hist_segment"
                )
                
                # Create segmented histogram with KDE - Reduced size from (10, 6) to (8, 4.5)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.histplot(data=filtered_df, x=num_var, hue=segment_var, kde=True, multiple="stack", ax=ax)
                plt.title(f"Distribution of {num_var.capitalize()} by {segment_var.capitalize()}")
                plt.xlabel(num_var.capitalize())
                plt.ylabel("Frequency")
                plt.legend(fontsize='small')
                plt.tight_layout()
            else:
                # Create simple histogram with KDE - Reduced size from (10, 6) to (8, 4.5)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.histplot(data=filtered_df, x=num_var, kde=True, ax=ax)
                plt.title(f"Distribution of {num_var.capitalize()}")
                plt.xlabel(num_var.capitalize())
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
                ["age", "bmi", "children", "charges"],
                key="box_var"
            )
            
            # Select categorical variable for grouping
            cat_var = st.selectbox(
                "Group by Category",
                ["region", "sex", "smoker"],
                key="box_cat"
            )
            
            # Create box plot - Reduced size from (10, 6) to (8, 4.5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=filtered_df, x=cat_var, y=num_var, ax=ax)
            plt.title(f"Box Plot of {num_var.capitalize()} by {cat_var.capitalize()}")
            plt.xlabel(cat_var.capitalize())
            plt.ylabel(num_var.capitalize())
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display summary statistics by group
            st.write("Summary Statistics by Group:")
            summary = filtered_df.groupby(cat_var)[num_var].describe()
            st.dataframe(summary, use_container_width=True)


