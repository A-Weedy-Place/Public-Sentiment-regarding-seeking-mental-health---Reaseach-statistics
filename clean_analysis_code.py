import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def load_and_clean_data(file_path):
    """Load and clean the dataset"""
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Remove timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    df.columns = [col.replace('\r\n', '') for col in df.columns]
    
    # Clean age groups by removing the word 'years' if present
    if 'Age Group' in df.columns:
        df['Age Group'] = df['Age Group'].str.replace(' years', '')
    
    # Fill missing ages based on age group
    age_group_mapping = {
        '10-19': 17,
        '20-29': 25,
        '30-39': 35,
        '40-49': 45,
        '50+': 55
    }
    
    # For each row with missing age, fill with the representative age for its age group
    if 'Age' in df.columns and 'Age Group' in df.columns:
        for idx, row in df[df['Age'].isna()].iterrows():
            age_group = row['Age Group']
            if age_group in age_group_mapping:
                df.at[idx, 'Age'] = age_group_mapping[age_group]
    
    # Handle age with a function to extract just the first number
    df['Age'] = df['Age'].apply(extract_first_number)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Handle missing values in Family history
    df['Family history of diagnosed mental illness in family'].fillna('None reported', inplace=True)
    
    # For Likert scale questions, fill with the mode (most common answer)
    null_counts = df.isnull().sum()
    likert_columns_with_nulls = [col for col in null_counts.index 
                                if null_counts[col] > 0 and col != 'Family history of diagnosed mental illness in family']
    
    # Fill each column with its mode
    for col in likert_columns_with_nulls:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
    
    return df

def extract_first_number(age_str):
    """Extract the first number from age string"""
    if pd.isna(age_str):
        return np.nan
    
    if isinstance(age_str, (int, float)):
        return age_str
    
    # Convert to string if it's not already
    age_str = str(age_str)
    
    # Extract the first number from the string
    import re
    numbers = re.findall(r'\d+', age_str)
    if numbers:
        return int(numbers[0])
    else:
        return np.nan

def convert_likert_to_numeric(df, columns, reverse=False):
    """Convert Likert scale responses to numeric values"""
    # Define the mapping for Likert scales
    if not reverse:
        likert_mapping = {
            'Strongly Disagree': 1, 
            'Disagree': 2, 
            'Agree': 3, 
            'Strongly agree': 4,
            'Strongly Agree': 4
        }
    else:
        # Reversed scoring
        likert_mapping = {
            'Strongly Disagree': 4, 
            'Disagree': 3, 
            'Agree': 2, 
            'Strongly agree': 1,
            'Strongly Agree': 1
        }
    
    # Apply mapping to each column
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(likert_mapping)
    
    return df

def calculate_scale_scores(df, exact_atspphs_openness, exact_atspphs_value, exact_dss_personal, exact_dss_perceived):
    """Calculate the scale scores for ATSPPHS and DSS"""
    df_numeric = df.copy()
    
    # Convert DSS items (higher score = more stigma)
    df_numeric = convert_likert_to_numeric(df_numeric, exact_dss_personal)
    df_numeric = convert_likert_to_numeric(df_numeric, exact_dss_perceived)
    
    # Check for the last item in perceived stigma that needs to be reversed
    reversed_item = [col for col in exact_dss_perceived if 'vote for a student' in col]
    if reversed_item:
        likert_mapping = {'Strongly Disagree': 4, 'Disagree': 3, 'Agree': 2, 'Strongly agree': 1, 'Strongly Agree': 1}
        df_numeric[reversed_item[0]] = df_numeric[reversed_item[0]].map(likert_mapping)
    
    # Convert ATSPPHS items
    df_numeric = convert_likert_to_numeric(df_numeric, exact_atspphs_openness)
    df_numeric = convert_likert_to_numeric(df_numeric, exact_atspphs_value, reverse=True)
    
    # Calculate scale scores
    # DSS Personal Scale
    df_numeric['DSS_Personal'] = df_numeric[exact_dss_personal].sum(axis=1)
    
    # DSS Perceived Scale
    df_numeric['DSS_Perceived'] = df_numeric[exact_dss_perceived].sum(axis=1)
    
    # DSS Total Score
    df_numeric['DSS_Total'] = df_numeric['DSS_Personal'] + df_numeric['DSS_Perceived']
    
    # ATSPPHS Openness Scale
    df_numeric['ATSPPHS_Openness'] = df_numeric[exact_atspphs_openness].sum(axis=1)
    
    # ATSPPHS Value Scale
    df_numeric['ATSPPHS_Value'] = df_numeric[exact_atspphs_value].sum(axis=1)
    
    # ATSPPHS Total Score
    df_numeric['ATSPPHS_Total'] = df_numeric['ATSPPHS_Openness'] + df_numeric['ATSPPHS_Value']
    
    # Create binary variables for the scales based on median split
    df_numeric['DSS_Personal_High'] = (df_numeric['DSS_Personal'] > df_numeric['DSS_Personal'].median()).astype(int)
    df_numeric['DSS_Perceived_High'] = (df_numeric['DSS_Perceived'] > df_numeric['DSS_Perceived'].median()).astype(int)
    df_numeric['DSS_Total_High'] = (df_numeric['DSS_Total'] > df_numeric['DSS_Total'].median()).astype(int)
    
    df_numeric['ATSPPHS_Openness_High'] = (df_numeric['ATSPPHS_Openness'] > df_numeric['ATSPPHS_Openness'].median()).astype(int)
    df_numeric['ATSPPHS_Value_High'] = (df_numeric['ATSPPHS_Value'] > df_numeric['ATSPPHS_Value'].median()).astype(int)
    df_numeric['ATSPPHS_Total_High'] = (df_numeric['ATSPPHS_Total'] > df_numeric['ATSPPHS_Total'].median()).astype(int)
    
    # Add a field grouping
    df_numeric['Field_Group'] = df['Field of Study'].apply(
        lambda x: 'Medicine/Nursing' if x == 'Medicine/ Nursing' else 'Other Fields'
    )
    
    return df_numeric

def mean_std(data):
    """Calculate mean and standard deviation formatted as a string"""
    return f"{data.mean():.2f} ± {data.std():.2f}"

def generate_demographic_table(df, df_numeric):
    """Generate Table 1: Demographic characteristics of participants"""
    # Calculate frequency and percentages for categorical variables
    demographic_vars = {
        'Gender': df['Gender'].value_counts(),
        'Living arrangement': df['Living arrangement'].value_counts(),
        'Marital status': df['Marital status'].value_counts(),
        'Family system': df['Family system'].value_counts(),
        'History of suicide in family': df['History of suicide in family'].value_counts(),
        'Family history of diagnosed mental illness in family': df['Family history of diagnosed mental illness in family'].value_counts()
    }
    
    # Mental health conditions
    mental_health_conditions = {
        'Depression': df['Depression'].value_counts(),
        'Generalized Anxiety Disorder (GAD)': df['GAD'].value_counts(),
        'Panic Disorder': df['Panic'].value_counts(),
        'Schizophrenia': df['Schiz'].value_counts(),
        'Bipolar Disorder': df['Bipolar'].value_counts(),
        'No diagnosed mental illness': df['None'].value_counts()
    }
    
    # Create a DataFrame for the demographics table
    demographics_table = pd.DataFrame(columns=['Characteristics', 'n (%)'])
    
    # Add age row
    age_row = pd.DataFrame({
        'Characteristics': ['Age in years, mean (SD)'],
        'n (%)': [f"{df_numeric['Age'].mean():.2f} ± {df_numeric['Age'].std():.2f}"]
    })
    demographics_table = pd.concat([demographics_table, age_row], ignore_index=True)
    
    # Add other demographic variables
    for var, counts in demographic_vars.items():
        for category, count in counts.items():
            percentage = count / len(df) * 100
            row = pd.DataFrame({
                'Characteristics': [f"{var}: {category}"],
                'n (%)': [f"{count} ({percentage:.1f}%)"]
            })
            demographics_table = pd.concat([demographics_table, row], ignore_index=True)
    
    # Add a section header for mental health conditions
    header_row = pd.DataFrame({
        'Characteristics': ["Mental Health Conditions"],
        'n (%)': [""]
    })
    demographics_table = pd.concat([demographics_table, header_row], ignore_index=True)
    
    # Add mental health conditions
    for condition, counts in mental_health_conditions.items():
        for response, count in counts.items():
            if response == 'yes':
                percentage = count / len(df) * 100
                row = pd.DataFrame({
                    'Characteristics': [f"{condition}"],
                    'n (%)': [f"{count} ({percentage:.1f}%)"]
                })
                demographics_table = pd.concat([demographics_table, row], ignore_index=True)
    
    print("Table 1: Demographic characteristics of participants (n=199)")
    print(demographics_table)
    
    return demographics_table

def generate_field_comparison_table(df, df_numeric):
    """Generate Table 2: Comparison of characteristics by field of study"""
    # List of characteristics to compare
    categorical_vars = ['Gender', 'Living arrangement', 'Marital status', 'Family system', 
                       'History of suicide in family', 'Depression']
    continuous_vars = ['Age']
    
    # Create an empty DataFrame for the comparison table
    comparison_table = pd.DataFrame(columns=['Characteristics', 'Medicine/Nursing', 'Engineering/IT', 'Arts/Humanities', 'p-value'])
    
    # Add rows for continuous variables (like age)
    for var in continuous_vars:
        # Calculate statistics for each group
        med_data = df_numeric[df['Field of Study'] == 'Medicine/ Nursing'][var]
        eng_data = df_numeric[df['Field of Study'] == 'Engineering/ IT'][var]
        arts_data = df_numeric[df['Field of Study'] == 'Arts/ Humanities'][var]
        
        # Calculate p-value using ANOVA
        groups = [med_data.dropna(), eng_data.dropna(), arts_data.dropna()]
        f_stat, p_val = stats.f_oneway(*groups)
        
        # Create a row for this variable
        row = pd.DataFrame({
            'Characteristics': [f"{var} in years, mean (SD)"],
            'Medicine/Nursing': [f"{med_data.mean():.1f} ± {med_data.std():.1f}"],
            'Engineering/IT': [f"{eng_data.mean():.1f} ± {eng_data.std():.1f}"],
            'Arts/Humanities': [f"{arts_data.mean():.1f} ± {arts_data.std():.1f}"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        comparison_table = pd.concat([comparison_table, row], ignore_index=True)
    
    # Add rows for categorical variables
    for var in categorical_vars:
        # Get the unique categories for this variable
        categories = df[var].unique()
        
        for category in categories:
            # Count occurrences in each group
            med_count = sum((df['Field of Study'] == 'Medicine/ Nursing') & (df[var] == category))
            med_total = sum(df['Field of Study'] == 'Medicine/ Nursing')
            med_pct = med_count / med_total * 100
            
            eng_count = sum((df['Field of Study'] == 'Engineering/ IT') & (df[var] == category))
            eng_total = sum(df['Field of Study'] == 'Engineering/ IT')
            eng_pct = eng_count / eng_total * 100
            
            arts_count = sum((df['Field of Study'] == 'Arts/ Humanities') & (df[var] == category))
            arts_total = sum(df['Field of Study'] == 'Arts/ Humanities')
            arts_pct = arts_count / arts_total * 100
            
            # Create a row for this category
            row = pd.DataFrame({
                'Characteristics': [f"{var}: {category}"],
                'Medicine/Nursing': [f"{med_count} ({med_pct:.1f}%)"],
                'Engineering/IT': [f"{eng_count} ({eng_pct:.1f}%)"],
                'Arts/Humanities': [f"{arts_count} ({arts_pct:.1f}%)"],
                'p-value': [""]  # We'll calculate p-values for the variable as a whole, not each category
            })
            
            # Add row to table
            comparison_table = pd.concat([comparison_table, row], ignore_index=True)
        
        # Calculate p-value for this categorical variable
        contingency_table = pd.crosstab(df[var], df['Field of Study'])
        
        # Use Chi-square test
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        # If any expected frequency is < 5, note that Fisher's exact test would be more appropriate
        test_type = "Chi2"
        if (expected < 5).any():
            test_type = "Fisher"
        
        # Update p-value in the last row for this variable
        idx = comparison_table[comparison_table['Characteristics'].str.contains(var)].index[-1]
        comparison_table.loc[idx, 'p-value'] = f"{p_val:.3f}{' *' if test_type == 'Fisher' else ''}"
    
    print("Table 2: Comparison of characteristics by field of study")
    print(comparison_table)
    
    # Add footnote for Fisher's exact test
    if any('*' in str(val) for val in comparison_table['p-value']):
        print("\n* Fisher's exact test would be more appropriate due to low expected frequencies")
    
    return comparison_table

def generate_scale_scores_table_improved(df_numeric):
    """Generate an improved Table 3: Comparison of scale scores across all three fields of study"""
    scales = ['ATSPPHS_Openness', 'ATSPPHS_Value', 'ATSPPHS_Total', 
             'DSS_Personal', 'DSS_Perceived', 'DSS_Total']
    
    # Create an empty DataFrame for the scale scores table
    scale_table = pd.DataFrame(columns=['Characteristics', 'Total', 'Medicine/Nursing', 'Engineering/IT', 'Arts/Humanities', 'p-value'])
    
    # Add rows for each scale
    for scale in scales:
        # Calculate overall mean and SD
        total_mean = df_numeric[scale].mean()
        total_sd = df_numeric[scale].std()
        
        # Calculate mean and SD for each field
        med_data = df_numeric[df_numeric['Field of Study'] == 'Medicine/ Nursing'][scale]
        med_mean = med_data.mean()
        med_sd = med_data.std()
        
        eng_data = df_numeric[df_numeric['Field of Study'] == 'Engineering/ IT'][scale]
        eng_mean = eng_data.mean()
        eng_sd = eng_data.std()
        
        arts_data = df_numeric[df_numeric['Field of Study'] == 'Arts/ Humanities'][scale]
        arts_mean = arts_data.mean()
        arts_sd = arts_data.std()
        
        # Calculate p-value using ANOVA (comparing all 3 groups)
        groups = [med_data, eng_data, arts_data]
        f_stat, p_val = stats.f_oneway(*groups)
        
        # Create a row for this scale
        row = pd.DataFrame({
            'Characteristics': [scale],
            'Total': [f"{total_mean:.2f} +/- {total_sd:.2f}"],
            'Medicine/Nursing': [f"{med_mean:.2f} +/- {med_sd:.2f}"],
            'Engineering/IT': [f"{eng_mean:.2f} +/- {eng_sd:.2f}"],
            'Arts/Humanities': [f"{arts_mean:.2f} +/- {arts_sd:.2f}"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        scale_table = pd.concat([scale_table, row], ignore_index=True)
    
    print("Table 3: Comparison of ATSPPHS and DSS scale scores across all fields of study")
    print(scale_table)
    
    return scale_table


def generate_scale_scores_table(df_numeric):
    """Generate Table 3: Comparison of ATSPPHS and DSS scale scores by field of study"""
    # Calculate mean and SD for each scale by field group
    scales = ['ATSPPHS_Openness', 'ATSPPHS_Value', 'ATSPPHS_Total', 
             'DSS_Personal', 'DSS_Perceived', 'DSS_Total']
    
    # Create an empty DataFrame for the scale scores table
    scale_table = pd.DataFrame(columns=['Characteristics', 'Total', 'Medicine/Nursing', 'Other Fields', 'p-value'])
    
    # Add rows for each scale
    for scale in scales:
        # Calculate overall mean and SD
        total_mean = df_numeric[scale].mean()
        total_sd = df_numeric[scale].std()
        
        # Calculate mean and SD for Medicine/Nursing
        med_data = df_numeric[df_numeric['Field_Group'] == 'Medicine/Nursing'][scale]
        med_mean = med_data.mean()
        med_sd = med_data.std()
        
        # Calculate mean and SD for Other Fields
        other_data = df_numeric[df_numeric['Field_Group'] == 'Other Fields'][scale]
        other_mean = other_data.mean()
        other_sd = other_data.std()
        
        # Calculate p-value for difference between groups
        _, p_val = stats.ttest_ind(med_data, other_data, equal_var=False)
        
        # Create a row for this scale
        row = pd.DataFrame({
            'Characteristics': [scale],
            'Total': [f"{total_mean:.2f} ± {total_sd:.2f}"],
            'Medicine/Nursing': [f"{med_mean:.2f} ± {med_sd:.2f}"],
            'Other Fields': [f"{other_mean:.2f} ± {other_sd:.2f}"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        scale_table = pd.concat([scale_table, row], ignore_index=True)
    
    print("Table 3: Comparison of ATSPPHS and DSS scale scores by field of study")
    print(scale_table)
    
    return scale_table

def generate_atspphs_distribution_table(df, exact_atspphs_openness, exact_atspphs_value):
    """Generate Table 4: Distribution of ATSPPHS items (agreement vs. disagreement)"""
    # Define agreement and disagreement values
    agree_values = ['Strongly agree', 'Strongly Agree', 'Agree']
    disagree_values = ['Strongly Disagree', 'Disagree']
    
    # Create an empty DataFrame for the ATSPPHS table
    atspphs_table = pd.DataFrame(columns=['S.No', 'Items', 'Agree n (%)', 'Disagree n (%)', 'p-value'])
    
    # Add a section header for Openness Scale
    header_row = pd.DataFrame({
        'S.No': [""],
        'Items': ["Openness Scale"],
        'Agree n (%)': [""],
        'Disagree n (%)': [""],
        'p-value': [""]
    })
    atspphs_table = pd.concat([header_row, atspphs_table], ignore_index=True)
    
    # Add rows for Openness Scale items
    for i, item in enumerate(exact_atspphs_openness, 1):
        # Calculate overall agreement/disagreement
        agree_count = df[item].isin(agree_values).sum()
        disagree_count = df[item].isin(disagree_values).sum()
        agree_pct = agree_count / len(df) * 100
        disagree_pct = disagree_count / len(df) * 100
        
        # Calculate p-value between field groups
        med_agree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(agree_values).sum()
        med_disagree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(disagree_values).sum()
        other_agree = agree_count - med_agree
        other_disagree = disagree_count - med_disagree
        
        contingency = np.array([[med_agree, med_disagree], [other_agree, other_disagree]])
        _, p_val, _, _ = stats.chi2_contingency(contingency)
        
        # Create a row for this item
        item_name = item
        if len(item_name) > 50:  # Truncate long item names
            item_name = item_name[:47] + "..."
            
        row = pd.DataFrame({
            'S.No': [i],
            'Items': [item_name],
            'Agree n (%)': [f"{agree_count} ({agree_pct:.1f}%)"],
            'Disagree n (%)': [f"{disagree_count} ({disagree_pct:.1f}%)"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        atspphs_table = pd.concat([atspphs_table, row], ignore_index=True)
    
    # Add a section header for Value Scale
    header_row = pd.DataFrame({
        'S.No': [""],
        'Items': ["Value Scale"],
        'Agree n (%)': [""],
        'Disagree n (%)': [""],
        'p-value': [""]
    })
    atspphs_table = pd.concat([atspphs_table, header_row], ignore_index=True)
    
    # Add rows for Value Scale items
    for i, item in enumerate(exact_atspphs_value, 1):
        # Calculate overall agreement/disagreement
        agree_count = df[item].isin(agree_values).sum()
        disagree_count = df[item].isin(disagree_values).sum()
        agree_pct = agree_count / len(df) * 100
        disagree_pct = disagree_count / len(df) * 100
        
        # Calculate p-value between field groups
        med_agree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(agree_values).sum()
        med_disagree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(disagree_values).sum()
        other_agree = agree_count - med_agree
        other_disagree = disagree_count - med_disagree
        
        contingency = np.array([[med_agree, med_disagree], [other_agree, other_disagree]])
        _, p_val, _, _ = stats.chi2_contingency(contingency)
        
        # Create a row for this item
        item_name = item
        if len(item_name) > 50:  # Truncate long item names
            item_name = item_name[:47] + "..."
            
        row = pd.DataFrame({
            'S.No': [i],
            'Items': [item_name],
            'Agree n (%)': [f"{agree_count} ({agree_pct:.1f}%)"],
            'Disagree n (%)': [f"{disagree_count} ({disagree_pct:.1f}%)"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        atspphs_table = pd.concat([atspphs_table, row], ignore_index=True)
    
    print("Table 4: Distribution of ATSPPHS items (agreement vs. disagreement)")
    print(atspphs_table)
    
    return atspphs_table

def generate_dss_distribution_table(df, exact_dss_personal, exact_dss_perceived):
    """Generate Table 5: Frequency of DSS items (agreement vs. disagreement)"""
    # Define agreement and disagreement values
    agree_values = ['Strongly agree', 'Strongly Agree', 'Agree']
    disagree_values = ['Strongly Disagree', 'Disagree']
    
    # Create an empty DataFrame for the DSS table
    dss_table = pd.DataFrame(columns=['S.No', 'Items', 'Agree n (%)', 'Disagree n (%)', 'p-value'])
    
    # Add a section header for Personal Stigma Scale
    header_row = pd.DataFrame({
        'S.No': [""],
        'Items': ["Personal Stigma Scale"],
        'Agree n (%)': [""],
        'Disagree n (%)': [""],
        'p-value': [""]
    })
    dss_table = pd.concat([header_row, dss_table], ignore_index=True)
    
    # Add rows for Personal Stigma Scale items
    for i, item in enumerate(exact_dss_personal, 1):
        # Calculate overall agreement/disagreement
        agree_count = df[item].isin(agree_values).sum()
        disagree_count = df[item].isin(disagree_values).sum()
        agree_pct = agree_count / len(df) * 100
        disagree_pct = disagree_count / len(df) * 100
        
        # Calculate p-value between field groups
        med_agree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(agree_values).sum()
        med_disagree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(disagree_values).sum()
        other_agree = agree_count - med_agree
        other_disagree = disagree_count - med_disagree
        
        contingency = np.array([[med_agree, med_disagree], [other_agree, other_disagree]])
        _, p_val, _, _ = stats.chi2_contingency(contingency)
        
        # Create a row for this item
        item_name = item
        if len(item_name) > 50:  # Truncate long item names
            item_name = item_name[:47] + "..."
            
        row = pd.DataFrame({
            'S.No': [i],
            'Items': [item_name],
            'Agree n (%)': [f"{agree_count} ({agree_pct:.1f}%)"],
            'Disagree n (%)': [f"{disagree_count} ({disagree_pct:.1f}%)"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        dss_table = pd.concat([dss_table, row], ignore_index=True)
    
    # Add a section header for Perceived Stigma Scale
    header_row = pd.DataFrame({
        'S.No': [""],
        'Items': ["Perceived Stigma Scale"],
        'Agree n (%)': [""],
        'Disagree n (%)': [""],
        'p-value': [""]
    })
    dss_table = pd.concat([dss_table, header_row], ignore_index=True)
    
    # Add rows for Perceived Stigma Scale items
    for i, item in enumerate(exact_dss_perceived, 1):
        # Calculate overall agreement/disagreement
        agree_count = df[item].isin(agree_values).sum()
        disagree_count = df[item].isin(disagree_values).sum()
        agree_pct = agree_count / len(df) * 100
        disagree_pct = disagree_count / len(df) * 100
        
        # Calculate p-value between field groups
        med_agree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(agree_values).sum()
        med_disagree = df[df['Field of Study'] == 'Medicine/ Nursing'][item].isin(disagree_values).sum()
        other_agree = agree_count - med_agree
        other_disagree = disagree_count - med_disagree
        
        contingency = np.array([[med_agree, med_disagree], [other_agree, other_disagree]])
        _, p_val, _, _ = stats.chi2_contingency(contingency)
        
        # Create a row for this item
        item_name = item
        if len(item_name) > 50:  # Truncate long item names
            item_name = item_name[:47] + "..."
            
        row = pd.DataFrame({
            'S.No': [i],
            'Items': [item_name],
            'Agree n (%)': [f"{agree_count} ({agree_pct:.1f}%)"],
            'Disagree n (%)': [f"{disagree_count} ({disagree_pct:.1f}%)"],
            'p-value': [f"{p_val:.3f}"]
        })
        
        # Add row to table
        dss_table = pd.concat([dss_table, row], ignore_index=True)
    
    print("Table 5: Frequency (percentage) of DSS items (agreement vs. disagreement)")
    print(dss_table)
    
    return dss_table



def run_regression_models_improved(df_numeric):
    """
    Run multiple regression models and generate Table 6 with proper crude and adjusted coefficients
    Based on reference paper variables of interest
    """
    import statsmodels.api as sm
    
    # Create a DataFrame for the predictors table
    predictors_table = pd.DataFrame(columns=['Outcome and Predictors', 'Crude β (95% CI)', 'p-value', 'Adjusted β (95% CI)', 'p-value'])
    
    # Add a header row for ATSPPHS model
    header_row = pd.DataFrame({
        'Outcome and Predictors': ["Attitude towards seeking help (ATSPPHS_Total)"],
        'Crude β (95% CI)': [""],
        'p-value': [""],
        'Adjusted β (95% CI)': [""],
        'p-value': [""]
    })
    predictors_table = pd.concat([predictors_table, header_row], ignore_index=True)
    
    # Define predictors for ATSPPHS model
    atspphs_predictors = ['Age', 'DSS_Personal', 'DSS_Perceived']
    atspphs_cat_vars = ['Gender', 'Living arrangement', 'Depression']
    
    # Add constant to df_numeric for crude models
    df_numeric = df_numeric.copy()
    df_numeric['const'] = 1.0
    
    # Run crude (univariate) models for ATSPPHS
    for var in atspphs_predictors:
        # Continuous predictors
        X = df_numeric[['const', var]].dropna()
        y = df_numeric.loc[X.index, 'ATSPPHS_Total']
        
        # Run simple regression
        crude_model = sm.OLS(y, X).fit()
        
        # Extract results
        coef = crude_model.params[var]
        p_val = crude_model.pvalues[var]
        conf_int = crude_model.conf_int().loc[var]
        
        # Create a row
        row = pd.DataFrame({
            'Outcome and Predictors': [var],
            'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
            'p-value': [f"{p_val:.3f}"],
            'Adjusted β (95% CI)': [""],
            'p-value': [""]
        })
        predictors_table = pd.concat([predictors_table, row], ignore_index=True)
    
    # Run crude models for categorical predictors of ATSPPHS
    for var in atspphs_cat_vars:
        if var == 'Gender':
            # Create dummy for Gender (Male vs Female)
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Gender_Male'] = (df_numeric['Gender'] == 'Male').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'ATSPPHS_Total']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Gender_Male']
            p_val = crude_model.pvalues['Gender_Male']
            conf_int = crude_model.conf_int().loc['Gender_Male']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Gender (Male)'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
            
        elif var == 'Living arrangement':
            # Create dummy for Living with family vs not
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'ATSPPHS_Total']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Living_with_family']
            p_val = crude_model.pvalues['Living_with_family']
            conf_int = crude_model.conf_int().loc['Living_with_family']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Currently living with family'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
            
        elif var == 'Depression':
            # Create dummy for Depression (yes vs no)
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Depression_yes'] = (df_numeric['Depression'] == 'yes').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'ATSPPHS_Total']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Depression_yes']
            p_val = crude_model.pvalues['Depression_yes']
            conf_int = crude_model.conf_int().loc['Depression_yes']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Previous history of depression'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
    
    # Run multiple regression model for ATSPPHS (adjusted model)
    X_adj = pd.DataFrame()
    X_adj['const'] = 1.0
    
    # Add continuous predictors
    for var in atspphs_predictors:
        X_adj[var] = df_numeric[var]
    
    # Add categorical predictors
    X_adj['Gender_Male'] = (df_numeric['Gender'] == 'Male').astype(float)
    X_adj['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
    X_adj['Depression_yes'] = (df_numeric['Depression'] == 'yes').astype(float)
    
    # Clean data
    X_adj = X_adj.dropna()
    y_adj = df_numeric.loc[X_adj.index, 'ATSPPHS_Total']
    
    # Run multiple regression
    adj_model = sm.OLS(y_adj, X_adj).fit()
    
    # Update rows with adjusted coefficients
    for i, row in predictors_table.iterrows():
        if row['Outcome and Predictors'] == 'Age':
            coef = adj_model.params['Age']
            p_val = adj_model.pvalues['Age']
            conf_int = adj_model.conf_int().loc['Age']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'DSS_Personal':
            coef = adj_model.params['DSS_Personal']
            p_val = adj_model.pvalues['DSS_Personal']
            conf_int = adj_model.conf_int().loc['DSS_Personal']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'DSS_Perceived':
            coef = adj_model.params['DSS_Perceived']
            p_val = adj_model.pvalues['DSS_Perceived']
            conf_int = adj_model.conf_int().loc['DSS_Perceived']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Gender (Male)':
            coef = adj_model.params['Gender_Male']
            p_val = adj_model.pvalues['Gender_Male']
            conf_int = adj_model.conf_int().loc['Gender_Male']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Currently living with family':
            coef = adj_model.params['Living_with_family']
            p_val = adj_model.pvalues['Living_with_family']
            conf_int = adj_model.conf_int().loc['Living_with_family']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Previous history of depression':
            coef = adj_model.params['Depression_yes']
            p_val = adj_model.pvalues['Depression_yes']
            conf_int = adj_model.conf_int().loc['Depression_yes']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
    
    # Add header for Personal Depression Stigma model
    header_row = pd.DataFrame({
        'Outcome and Predictors': ["Personal Depression Stigma (DSS_Personal)"],
        'Crude β (95% CI)': [""],
        'p-value': [""],
        'Adjusted β (95% CI)': [""],
        'p-value': [""]
    })
    predictors_table = pd.concat([predictors_table, header_row], ignore_index=True)
    
    # Define predictors for DSS_Personal model
    dss_personal_predictors = ['Age', 'ATSPPHS_Total']
    dss_personal_cat_vars = ['Gender', 'Living arrangement', 'Depression']
    
    # Run crude (univariate) models for DSS_Personal
    for var in dss_personal_predictors:
        # Continuous predictors
        X = df_numeric[['const', var]].dropna()
        y = df_numeric.loc[X.index, 'DSS_Personal']
        
        # Run simple regression
        crude_model = sm.OLS(y, X).fit()
        
        # Extract results
        coef = crude_model.params[var]
        p_val = crude_model.pvalues[var]
        conf_int = crude_model.conf_int().loc[var]
        
        # Rename ATSPPHS_Total to more readable name
        var_name = var
        if var == 'ATSPPHS_Total':
            var_name = 'Attitude towards seeking help'
            
        # Create a row
        row = pd.DataFrame({
            'Outcome and Predictors': [var_name],
            'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
            'p-value': [f"{p_val:.3f}"],
            'Adjusted β (95% CI)': [""],
            'p-value': [""]
        })
        predictors_table = pd.concat([predictors_table, row], ignore_index=True)
    
    # Run crude models for categorical predictors of DSS_Personal
    for var in dss_personal_cat_vars:
        if var == 'Gender':
            # Create dummy for Gender (Male vs Female)
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Gender_Male'] = (df_numeric['Gender'] == 'Male').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'DSS_Personal']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Gender_Male']
            p_val = crude_model.pvalues['Gender_Male']
            conf_int = crude_model.conf_int().loc['Gender_Male']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Gender (Male)'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
            
        elif var == 'Living arrangement':
            # Create dummy for Living with family vs not
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'DSS_Personal']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Living_with_family']
            p_val = crude_model.pvalues['Living_with_family']
            conf_int = crude_model.conf_int().loc['Living_with_family']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Currently living with family'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
            
        elif var == 'Depression':
            # Create dummy for Depression (yes vs no)
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Depression_yes'] = (df_numeric['Depression'] == 'yes').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'DSS_Personal']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Depression_yes']
            p_val = crude_model.pvalues['Depression_yes']
            conf_int = crude_model.conf_int().loc['Depression_yes']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Previous history of depression'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
    
    # Run multiple regression model for DSS_Personal (adjusted model)
    X_adj = pd.DataFrame()
    X_adj['const'] = 1.0
    
    # Add continuous predictors
    for var in dss_personal_predictors:
        X_adj[var] = df_numeric[var]
    
    # Add categorical predictors
    X_adj['Gender_Male'] = (df_numeric['Gender'] == 'Male').astype(float)
    X_adj['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
    X_adj['Depression_yes'] = (df_numeric['Depression'] == 'yes').astype(float)
    
    # Clean data
    X_adj = X_adj.dropna()
    y_adj = df_numeric.loc[X_adj.index, 'DSS_Personal']
    
    # Run multiple regression
    adj_model = sm.OLS(y_adj, X_adj).fit()
    
    # Update rows with adjusted coefficients
    for i, row in predictors_table.iterrows():
        if row['Outcome and Predictors'] == 'Age':
            if i > (predictors_table['Outcome and Predictors'] == 'Personal Depression Stigma (DSS_Personal)').idxmax():
                coef = adj_model.params['Age']
                p_val = adj_model.pvalues['Age']
                conf_int = adj_model.conf_int().loc['Age']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Attitude towards seeking help':
            coef = adj_model.params['ATSPPHS_Total']
            p_val = adj_model.pvalues['ATSPPHS_Total']
            conf_int = adj_model.conf_int().loc['ATSPPHS_Total']
            predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
            predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Gender (Male)':
            if i > (predictors_table['Outcome and Predictors'] == 'Personal Depression Stigma (DSS_Personal)').idxmax():
                coef = adj_model.params['Gender_Male']
                p_val = adj_model.pvalues['Gender_Male']
                conf_int = adj_model.conf_int().loc['Gender_Male']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Currently living with family':
            if i > (predictors_table['Outcome and Predictors'] == 'Personal Depression Stigma (DSS_Personal)').idxmax():
                coef = adj_model.params['Living_with_family']
                p_val = adj_model.pvalues['Living_with_family']
                conf_int = adj_model.conf_int().loc['Living_with_family']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Previous history of depression':
            if i > (predictors_table['Outcome and Predictors'] == 'Personal Depression Stigma (DSS_Personal)').idxmax():
                coef = adj_model.params['Depression_yes']
                p_val = adj_model.pvalues['Depression_yes']
                conf_int = adj_model.conf_int().loc['Depression_yes']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
    
    # Add header for Perceived Depression Stigma model
    header_row = pd.DataFrame({
        'Outcome and Predictors': ["Perceived Depression Stigma (DSS_Perceived)"],
        'Crude β (95% CI)': [""],
        'p-value': [""],
        'Adjusted β (95% CI)': [""],
        'p-value': [""]
    })
    predictors_table = pd.concat([predictors_table, header_row], ignore_index=True)
    
    # Define predictors for DSS_Perceived model
    dss_perceived_predictors = ['Age', 'ATSPPHS_Total']
    dss_perceived_cat_vars = ['Gender', 'Living arrangement', 'Depression']
    
    # Run crude (univariate) models for DSS_Perceived
    for var in dss_perceived_predictors:
        # Continuous predictors
        X = df_numeric[['const', var]].dropna()
        y = df_numeric.loc[X.index, 'DSS_Perceived']
        
        # Run simple regression
        crude_model = sm.OLS(y, X).fit()
        
        # Extract results
        coef = crude_model.params[var]
        p_val = crude_model.pvalues[var]
        conf_int = crude_model.conf_int().loc[var]
        
        # Rename ATSPPHS_Total to more readable name
        var_name = var
        if var == 'ATSPPHS_Total':
            var_name = 'Attitude towards seeking help'
            
        # Create a row
        row = pd.DataFrame({
            'Outcome and Predictors': [var_name],
            'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
            'p-value': [f"{p_val:.3f}"],
            'Adjusted β (95% CI)': [""],
            'p-value': [""]
        })
        predictors_table = pd.concat([predictors_table, row], ignore_index=True)
    
    # Run crude models for categorical predictors of DSS_Perceived
    for var in dss_perceived_cat_vars:
        if var == 'Living arrangement':
            # Create dummy for Living with family vs not
            X = pd.DataFrame()
            X['const'] = 1.0
            X['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
            X = X.dropna()
            y = df_numeric.loc[X.index, 'DSS_Perceived']
            
            # Run simple regression
            crude_model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = crude_model.params['Living_with_family']
            p_val = crude_model.pvalues['Living_with_family']
            conf_int = crude_model.conf_int().loc['Living_with_family']
            
            # Create a row
            row = pd.DataFrame({
                'Outcome and Predictors': ['Currently living with family'],
                'Crude β (95% CI)': [f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"],
                'p-value': [f"{p_val:.3f}"],
                'Adjusted β (95% CI)': [""],
                'p-value': [""]
            })
            predictors_table = pd.concat([predictors_table, row], ignore_index=True)
            
    # Run multiple regression model for DSS_Perceived (adjusted model)
    X_adj = pd.DataFrame()
    X_adj['const'] = 1.0
    
    # Add continuous predictors
    for var in dss_perceived_predictors:
        X_adj[var] = df_numeric[var]
    
    # Add categorical predictors
    X_adj['Living_with_family'] = (df_numeric['Living arrangement'] == 'Living with family').astype(float)
    
    # Clean data
    X_adj = X_adj.dropna()
    y_adj = df_numeric.loc[X_adj.index, 'DSS_Perceived']
    
    # Run multiple regression
    adj_model = sm.OLS(y_adj, X_adj).fit()
    
    # Update rows with adjusted coefficients
    for i, row in predictors_table.iterrows():
        if row['Outcome and Predictors'] == 'Age':
            if i > (predictors_table['Outcome and Predictors'] == 'Perceived Depression Stigma (DSS_Perceived)').idxmax():
                coef = adj_model.params['Age']
                p_val = adj_model.pvalues['Age']
                conf_int = adj_model.conf_int().loc['Age']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Attitude towards seeking help':
            if i > (predictors_table['Outcome and Predictors'] == 'Perceived Depression Stigma (DSS_Perceived)').idxmax():
                coef = adj_model.params['ATSPPHS_Total']
                p_val = adj_model.pvalues['ATSPPHS_Total']
                conf_int = adj_model.conf_int().loc['ATSPPHS_Total']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
        elif row['Outcome and Predictors'] == 'Currently living with family':
            if i > (predictors_table['Outcome and Predictors'] == 'Perceived Depression Stigma (DSS_Perceived)').idxmax():
                coef = adj_model.params['Living_with_family']
                p_val = adj_model.pvalues['Living_with_family']
                conf_int = adj_model.conf_int().loc['Living_with_family']
                predictors_table.loc[i, 'Adjusted β (95% CI)'] = f"{coef:.2f} ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
                predictors_table.loc[i, 'p-value.1'] = f"{p_val:.3f}"
    
    print("Table 6: Predictors of attitude towards seeking help, and personal and perceived depression stigma")
    print(predictors_table)
    
    return predictors_table


def generate_mental_health_comparison(df):
    """
    Generate a comparison table of mental health conditions by field of study
    
    This function only focuses on mental health conditions and shows percentages
    of 'yes' responses across different fields of study.
    """
    # Mental health conditions to include
    mental_health_vars = ['Depression', 'GAD', 'Panic', 'Schiz', 'Bipolar', 'None']
    
    # Create an empty DataFrame for the table
    mh_table = pd.DataFrame(columns=['Mental Health Condition', 'Medicine/Nursing', 'Engineering/IT', 'Arts/Humanities', 'p-value'])
    
    # Add rows for mental health conditions (only showing "yes" responses)
    for var in mental_health_vars:
        # Count "yes" responses in each group
        med_count = sum((df['Field of Study'] == 'Medicine/ Nursing') & (df[var] == 'yes'))
        med_total = sum(df['Field of Study'] == 'Medicine/ Nursing')
        med_pct = med_count / med_total * 100
        
        eng_count = sum((df['Field of Study'] == 'Engineering/ IT') & (df[var] == 'yes'))
        eng_total = sum(df['Field of Study'] == 'Engineering/ IT')
        eng_pct = eng_count / eng_total * 100
        
        arts_count = sum((df['Field of Study'] == 'Arts/ Humanities') & (df[var] == 'yes'))
        arts_total = sum(df['Field of Study'] == 'Arts/ Humanities')
        arts_pct = arts_count / arts_total * 100
        
        # Create a nice display name for the condition
        display_name = var
        if var == 'GAD':
            display_name = 'Generalized Anxiety Disorder'
        elif var == 'Schiz':
            display_name = 'Schizophrenia'
        elif var == 'None':
            display_name = 'No diagnosed mental illness'
        elif var == 'Panic':
            display_name = 'Panic Disorder'
        
        # Create a row for this condition
        row = pd.DataFrame({
            'Mental Health Condition': [display_name],
            'Medicine/Nursing': [f"{med_count} ({med_pct:.1f}%)"],
            'Engineering/IT': [f"{eng_count} ({eng_pct:.1f}%)"],
            'Arts/Humanities': [f"{arts_count} ({arts_pct:.1f}%)"],
            'p-value': [""]
        })
        
        # Add row to table
        mh_table = pd.concat([mh_table, row], ignore_index=True)
        
        # Calculate p-value for this mental health condition
        contingency_table = pd.crosstab(df[var], df['Field of Study'])
        
        # Use Chi-square test
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        # If any expected frequency is < 5, note that Fisher's exact test would be more appropriate
        test_type = "Chi2"
        if (expected < 5).any():
            test_type = "Fisher"
        
        # Update p-value in the current row
        idx = mh_table.index[-1]
        mh_table.loc[idx, 'p-value'] = f"{p_val:.3f}{' *' if test_type == 'Fisher' else ''}"
    
    print("Table: Comparison of Mental Health Conditions by Field of Study")
    print(mh_table)
    
    # Add footnote for Fisher's exact test
    if any('*' in str(val) for val in mh_table['p-value']):
        print("\n* Fisher's exact test would be more appropriate due to low expected frequencies")
    
    return mh_table


def main():
    """Main function to execute the analysis and generate tables"""
    
    # Define the file path to your data
    file_path = 'data.xlsx'  # Replace with your actual file path
    
    # Load and clean the data
    df = load_and_clean_data(file_path)
    
    # Define the scale items
    # ATSPPHS Openness Subscale (5 items)
    atspphs_openness = [
        'If I believed I was having a mental breakdown, my first inclination would be to get professional help',
        'If I were experiencing a serious emotional crisis at this point in my life, I would be confident that I could find relief in psychotherapy',
        'I would want to get psychiatric attention if I was worried or upset for a long period of time',
        'At some future time, I might want to have psychological counselling',
        'A person with an emotional problem is not likely to solve it alone; he is likely to solve with professional help'
    ]

    # ATSPPHS Value Subscale (5 items)
    atspphs_value = [
        'The idea of talking about problems with a psychologist strikes me as a poor way to get rid of emotional conflicts',
        'There is something admirable in the attitude of a person who is willing to deal with own conflicts and fears without resorting to professional help',
        'Considering the time and expense involved in psychotherapy, it would have doubtful value for a person like me',
        'A person should work out one\'s own problems; getting psychological counselling should be the last resort',
        'Emotional difficulties, like many things, tend to work out by themselves'
    ]

    # DSS Personal Subscale (9 items)
    dss_personal = [
        'People with depression could snap out of it if they wanted.',
        'Depression is a sign of personal weakness',
        'Depression is not a real medical illness',
        'People with depression are dangerous',
        'It is best to avoid people with depression, so you do not become depressed yourself',
        'People with depression are unpredictable',
        'If I had depression, I would not tell anyone',
        'I would not study/mingle with someone if I knew they were depressed',
        'I would not vote for a student for a leadership role if I knew they were depressed'
    ]

    # DSS Perceived Subscale (9 items)
    dss_perceived = [
        'Most people believe that people with depression could snap out of it if they wanted',
        'Most people believe that depression is a sign of personal weakness',
        'Most people believe that depression is not a real medical illness',
        'Most people believe that people with depression are dangerous',
        'Most people believe it is best to avoid people with depression, so you do not become depressed yourself',
        'Most people believe that people with depression are unpredictable',
        'If they had depression, most people would not tell anyone',
        'Most people would not study/mingle with someone they knew were depressed',
        'Most people would vote for a student for a leadership role who they knew was depressed'
    ]
    
    
        # After defining all the scale items
    print("All columns in the dataset:")
    for col in df.columns:
        print(col)
        
    # Then check for partial matches for the missing items
    missing_value_items = [
        'There is something admirable in the attitude of a person who is willing to deal with own conflicts and fears without resorting to professional help',
        'A person should work out one\'s own problems; getting psychological counselling should be the last resort'
    ]

    for item in missing_value_items:
        print(f"\nLooking for: {item}")
        for col in df.columns:
            if any(word in col.lower() for word in ['admirable', 'attitude', 'work out', 'resort']):
                print(f"Potential match: {col}")
    
    # Find the exact column names that match our needed items
    exact_atspphs_openness = [col for col in df.columns if any(item.strip() in col.strip() for item in atspphs_openness)]
    exact_atspphs_value = [col for col in df.columns if any(item.strip() in col.strip() for item in atspphs_value)]
    exact_dss_personal = [col for col in df.columns if any(item.strip() in col.strip() for item in dss_personal)]
    exact_dss_perceived = [col for col in df.columns if any(item.strip() in col.strip() for item in dss_perceived)]
    
    print(f"ATSPPHS Openness items found: {len(exact_atspphs_openness)}")
    print(f"ATSPPHS Value items found: {len(exact_atspphs_value)}")
    print(f"DSS Personal items found: {len(exact_dss_personal)}")
    print(f"DSS Perceived items found: {len(exact_dss_perceived)}")
    
    # Calculate scale scores
    df_numeric = calculate_scale_scores(df, exact_atspphs_openness, exact_atspphs_value, exact_dss_personal, exact_dss_perceived)
    
    # Generate tables
    table1 = generate_demographic_table(df, df_numeric)
    mental_health_table = generate_mental_health_comparison(df)
    
    # Using the new mental health comparison table instead of the original table2
    table3 = generate_scale_scores_table_improved(df_numeric)
    table4 = generate_atspphs_distribution_table(df, exact_atspphs_openness, exact_atspphs_value)
    table5 = generate_dss_distribution_table(df, exact_dss_personal, exact_dss_perceived)
    table6 = run_regression_models_improved(df_numeric)
    
    # Return all tables for potential export or further analysis
    return {
        'table1': table1,
        'mental_health_table': mental_health_table,
        'table3': table3,
        'table4': table4,
        'table5': table5,
        'table6': table6,
        'df': df,
        'df_numeric': df_numeric
    }

if __name__ == "__main__":
    # Execute the main function
    results = main()
    
    # Optionally, save tables to CSV files
    # results['table1'].to_csv('table1_demographics.csv', index=False)
    # results['mental_health_table'].to_csv('mental_health_comparison.csv', index=False)
    # results['table3'].to_csv('table3_scale_scores.csv', index=False)
    # results['table4'].to_csv('table4_atspphs_distribution.csv', index=False)
    # results['table5'].to_csv('table5_dss_distribution.csv', index=False)
    results['table6'].to_csv('table6_predictors.csv', index=False)