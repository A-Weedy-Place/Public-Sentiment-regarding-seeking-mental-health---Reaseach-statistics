## (Use .py (python file) the jupiternotebook has a lot of exploration and redundant code)




# Mental Health Stigma Data Analysis

## Overview

This repository contains the code and analysis for examining the relationship between depression stigma and attitudes toward seeking professional psychological help. The analysis focuses on identifying predictors of stigma and help-seeking attitudes using regression techniques and assessing data quality through various psychometric approaches.

## Requirements

- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - scipy
  - statsmodels
  - matplotlib
  - seaborn
  - factor_analyzer

## Data Structure

The analysis uses data from a survey with 199 participants, containing:
- Demographic information (gender, age, field of study, etc.)
- Depression Stigma Scale (DSS) responses (18 items)
- Attitude Toward Seeking Professional Psychological Help Scale (ATSPPHS) responses (10 items)

## Code Structure

### 1. Data Loading and Preprocessing

- Loading Excel data
- Handling missing values
- Data type conversion
- Variable creation

### 2. Scale Reliability Analysis

- Cronbach's alpha calculation for:
  - DSS-Personal subscale
  - DSS-Perceived subscale
  - ATSPPHS-Openness subscale
  - ATSPPHS-Value subscale

### 3. Common Method Bias Assessment

- Harman's Single Factor Test
- Factor analysis using factor_analyzer
- Variance analysis

### 4. Response Pattern Analysis

- Detection of straight-line responding
- Extreme response patterns
- Mid-point response patterns

### 5. Correlation Analysis

- Spearman correlations between theoretically contradictory items
- Social desirability bias detection

### 6. Regression Analysis

- Univariate (crude) linear regression
- Individual predictor models
- Coefficient estimation with 95% confidence intervals
- Identification of significant predictors

### 7. Data Visualization

- Distribution plots for scale scores
- Correlation heatmaps
- Group comparison visualizations

## Key Technical Approaches

1. **Missing Data Handling**: Targeted imputation based on variable type (mode for categorical, median for continuous)

2. **Scale Construction**: Conversion of Likert responses to numeric values with appropriate reverse coding

3. **Robust Regression**: Individual regression models to avoid multicollinearity issues

4. **Data Quality Assessment**: Multi-faceted approach including reliability analysis, common method bias testing, and response pattern analysis

5. **Statistical Testing**: Parametric tests (t-tests, regression) with appropriate handling of assumptions

## Usage

1. Clone the repository
2. Install required dependencies
3. Place the data file (`data.xlsx`) in the project directory
4. Run the Jupyter notebooks in sequence:
   - `01_data_preparation.ipynb`
   - `02_reliability_analysis.ipynb`
   - `03_bias_detection.ipynb`
   - `04_regression_analysis.ipynb`

## Results

The analysis produces:
- Cronbach's alpha values for all subscales
- Common method bias assessment
- Response pattern statistics
- Regression coefficients with confidence intervals
- Tables and visualizations for reporting
