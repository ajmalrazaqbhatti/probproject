# Punjab Rice and Wheat Production Analysis Dashboard

## Project Overview

This dashboard analyzes agricultural production data in Punjab, India, focusing on:

- Rice and wheat production across 22 districts (1997-2019)
- Yield and production trends
- Statistical patterns and relationships
- Predictive modeling using regression

## Setup Instructions

### Option 1: Using setup script (recommended)

1. Make the setup script executable:

   ```
   chmod +x setup.sh
   ```

2. Run the setup script:

   ```
   ./setup.sh
   ```

3. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

4. Run the application:

   ```
   streamlit run main.py
   ```

   Or use the run script:

   ```
   ./run.sh
   ```

### Option 2: Manual setup

1. Create a virtual environment:

   ```
   python3 -m venv venv
   ```

2. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

3. Install requirements:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run main.py
   ```

## Dataset Description

**Source:** Punjab's Department of Agriculture and Farmers' Welfare

**Key Variables:**

- **District:** 22 administrative districts across Punjab
- **Crop:** Rice and Wheat (80% of Punjab's cultivated area)
- **Crop_Year:** 1997-2019 (over two decades)
- **Area:** Land under cultivation (hectares)
- **Production:** Total output (tonnes)
- **Yield:** Efficiency metric (tonnes/hectare)

**Size:** 1,035 observations with 6 variables

## Agricultural Context

This dataset represents Punjab's critical role in India's food security system. As India's agricultural powerhouse, Punjab contributes approximately:

- 19% of India's wheat production
- 11% of India's rice production

The data captures Punjab's agricultural intensification following the Green Revolution, with the state achieving some of the highest cereal yields in India. The yield improvements visible in this dataset represent the outcomes of significant investments in:

- Irrigation infrastructure (Punjab has >98% irrigated agriculture)
- High-yielding varieties
- Fertilizer and pesticide application
- Mechanization of farming operations

## Libraries Used

This project utilizes several Python libraries for data analysis and visualization:

| Library          | Purpose                                                      |
| ---------------- | ------------------------------------------------------------ |
| **Streamlit**    | Creates the interactive web application interface            |
| **Pandas**       | Data manipulation, cleaning, and analysis                    |
| **NumPy**        | Numerical computations and array operations                  |
| **Matplotlib**   | Core plotting and visualization functionality                |
| **Seaborn**      | Statistical data visualization with enhanced aesthetics      |
| **SciPy**        | Statistical functions, distributions, and hypothesis testing |
| **Scikit-learn** | Machine learning for regression modeling and evaluation      |
| **Base64**       | Encoding SVG files for display in the dashboard              |

## Statistical Methods Applied

### 1. Descriptive Statistics

**Where Applied:** "Descriptive Statistics" tab

**Concepts:**

- **Measures of Central Tendency:** Mean, median, mode

  - Example: Average rice yield across Punjab is ~3.98 tonnes/hectare

- **Measures of Dispersion:** Standard deviation, variance, range, IQR

  - Purpose: Quantify variability in agricultural outcomes

- **Percentiles:** Used to understand data distribution
  - Application: Identifying performance benchmarks across districts

### 2. Data Visualization with Statistical Context

**Where Applied:** "Graphical Analysis" tab

**Methods:**

- **Bar Charts with Confidence Intervals (95%):**

  - Statistical Concept: Inferential statistics, sampling error
  - Purpose: Showing estimate precision and statistical significance

- **Box Plots:**

  - Statistical Concept: Five-number summary (min, Q1, median, Q3, max)
  - Application: Visualizing outliers and district comparisons

- **Histograms with Kernel Density Estimation:**
  - Statistical Concept: Empirical distribution estimation
  - Purpose: Visualizing underlying data distribution

### 3. Contingency Tables and Cross-tabulation

**Where Applied:** "Descriptive Statistics" tab, Categorical Variables section

**Concepts:**

- **Frequency Distribution:** Counting categorical occurrences

  - Application: Understanding district and crop distributions

- **Contingency Tables:** Examining relationships between categories

  - Purpose: Analyzing how crop patterns vary across districts

- **Herfindahl-Hirschman Index:** Measure of concentration
  - Application: Quantifying concentration in agricultural production

## Probability Concepts Applied

### 1. Normal Distribution Analysis

**Where Applied:** "Probability Methods" tab

**Concepts:**

- **Distribution Fitting:**

  - Application: Modeling natural variation in crop yields

- **Probability Density Function (PDF):**

  - Purpose: Visualizing probability distribution of yields

- **Cumulative Distribution Function (CDF):**
  - Application: Calculating probability of yield being below threshold
  - Example: P(Yield < 4.0 tonnes/hectare) = 0.6327

### 2. Goodness-of-Fit Testing

**Where Applied:** "Probability Methods" tab

**Concepts:**

- **Kolmogorov-Smirnov Test:**

  - Purpose: Testing if yield data follows Normal distribution
  - Interpretation: p < 0.05 rejects normality assumption

- **Distribution Shape Analysis:**
  - Skewness: Measuring distribution asymmetry
  - Kurtosis: Analyzing tail behavior vs. Normal distribution

### 3. Percentile/Quantile Calculations

**Where Applied:** "Probability Methods" tab

**Concepts:**

- **Inverse CDF/Quantile Function:**

  - Application: Finding yield thresholds for percentiles
  - Example: "90% of yields are below 5.12 tonnes/hectare"

- **Empirical CDF:**
  - Purpose: Comparing observed distribution with theoretical

## Regression Analysis

**Where Applied:** "Regression Modeling" tab

**Statistical Concepts:**

- **Linear Regression:**

  - Purpose: Modeling relationship between Area/Year and Yield/Production

- **Train-Test Split:**

  - Statistical Concept: Cross-validation to assess model generalizability

- **Model Evaluation Metrics:**
  - R-squared: Measuring variance explained
  - RMSE: Quantifying prediction error

## Key Insights from Analysis

### Production Patterns

1. **Production Volume**:

   - Wheat generally shows higher production figures than rice
   - Highest production districts: Sangrur, Ludhiana, Patiala
   - Total production shows gradual increase over the 23-year period

2. **Cultivation Area**:

   - Total area under both crops has remained relatively stable
   - Some districts show shift from wheat to rice cultivation over time

3. **Yield Performance**:
   - Average wheat yield (~4.6 tonnes/hectare) exceeds rice yield (~3.9 tonnes/hectare)
   - Yield improved significantly over the 23-year period
   - Highest rice yields: Sangrur district (reaching 5.08 tonnes/hectare)
   - Highest wheat yields: Moga and Sangrur districts (exceeding 5.7 tonnes/hectare)

### Geographic Patterns

1. **District Variations**:

   - Central Punjab districts consistently outperform border districts
   - Up to 1.7x difference in yields between highest and lowest performing districts

2. **Regional Clustering**:
   - Districts show geographic clustering in performance
   - Malwa region (southern Punjab) shows differential performance from Doaba/Majha regions

### Temporal Trends

1. **Yield Evolution**:

   - Both crops show steady yield improvements from 1997-2019
   - Wheat yields increased by 28%, rice yields by 21%
   - Most significant improvement period: 2007-2012

2. **Year-to-Year Fluctuations**:
   - Notable yield drops in 2004, 2009, and 2014 (likely weather-related)
   - Exceptional performance years: 2011, 2016, 2018

## Limitations of the Dataset

1. **Crop Limitation**: Only includes rice and wheat, excluding other crops grown in Punjab
2. **Input Data Absence**: No information on fertilizer use, irrigation, or other inputs
3. **Economic Information**: Lacks price data and economic outcomes
4. **Environmental Factors**: No climate or weather data to explain yield fluctuations
5. **Limited Variables**: Missing potentially relevant factors like soil quality, irrigation source, etc.

## Running the Application

To run the dashboard:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run the application: `streamlit run main.py`
3. The dashboard will open in your default web browser

## Advanced Statistical Concepts

### Skewness and Kurtosis

- Skewness measures distribution asymmetry. Positive skew (right tail) in production data (0.78) indicated some districts with exceptionally high production.
- Kurtosis measures "tailedness" compared to Normal distribution. Higher kurtosis in rice yield (1.2) versus wheat (0.4) showed greater frequency of extreme values for rice.

### Heteroscedasticity

- Heteroscedasticity occurs when variance of errors varies across values of an independent variable.
- We observed this in Area-Production relationships, where larger areas showed greater variability in production.
- This violates an assumption of ordinary least squares regression and may affect standard errors.

### Future Methodological Improvements

Future improvements could include:

1. Multiple regression to incorporate more predictors simultaneously
2. Time series analysis for better temporal modeling
3. Hierarchical/mixed-effect models to account for district-level differences
4. Non-parametric methods for variables not following Normal distribution

## Project Structure

- `main.py`: Main Streamlit application
- `insurance.csv`: Dataset for analysis
- `requirements.txt`: Python dependencies

## Development

- Make sure to activate the virtual environment before development
- Add new dependencies to requirements.txt as needed
