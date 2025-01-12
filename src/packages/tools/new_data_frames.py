import numpy as np
import re
import pandas as pd
def remove_outliers(data):
    """
    Replace outliers (values outside 1.5 times the IQR) with NaN.
    """
    # Make a copy of the data to preserve the original dataset
    data = data.copy()

    # Iterate through numeric columns only
    for column in data.select_dtypes(include=[np.number]):
        # Calculate the first and third quartiles (25th and 75th percentiles)
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        # Compute the interquartile range
        IQR = Q3 - Q1

        # Determine outlier thresholds
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # Replace outliers with NaN using a lambda function
        data[column] = data[column].apply(lambda x: np.nan if x < lower_limit or x > upper_limit else x)

    # Return the modified dataset
    return data

def meaning_the_sessions(data):
    """
    Group session-based columns (e.g., '_session1', '_session2') and calculate row-wise means.
    Creates new columns with the suffix '_mean' for each group.
    """
    columns = data.columns
    identical_columns_dict = {}

    # Strip numerical session suffixes from column names (e.g., 'var1_session1' -> 'var1_session')
    stripped_columns = [re.sub(r'\d$', '', col) for col in columns if col != 'sex']
    unique_columns = set(stripped_columns)

    # Identify and group columns based on their base names
    for column_uni in unique_columns:
        matching_columns = [col for col in columns if re.sub(r'\d$', '', col) == column_uni]
        identical_columns_dict[column_uni] = matching_columns
        
        # Filter out the 'sex' column and columns with all NaN values
        matching_columns = [col for col in matching_columns if col != 'sex' and not data[col].isnull().all()]
        
        if not matching_columns:
            # Skip processing if no valid matching columns remain
            continue
        
        # Convert matching columns to numeric, ignoring errors
        numeric_data = data[matching_columns].apply(pd.to_numeric, errors='coerce')
        
        if numeric_data.empty:
            # Skip processing if no numeric data is available
            continue
        
        # Compute the row-wise mean while ignoring NaN values
        row_means = numeric_data.mean(axis=1, skipna=True)
        data[column_uni + '_mean'] = row_means

    # Return the updated dataset with new mean columns
    return data

def separating_genders(data):
    """
    Split the dataset into two subsets: one for males and one for females.
    Ensures the 'sex' column exists and is properly formatted.
    """
    # Check if the 'sex' column exists in the dataset
    if 'sex' not in data.columns:
        raise KeyError("'sex' column is missing in the DataFrame.")
    
    # Standardize the 'sex' column by stripping whitespace and converting to lowercase
    data['sex'] = data['sex'].astype(str).str.strip().str.lower()

    # Filter rows where 'sex' is 'm' (male) and create a new dataset
    new_df_male = data[data['sex'] == 'm'].copy()

    # Filter rows where 'sex' is 'f' (female) and create another new dataset
    new_df_female = data[data['sex'] == 'f'].copy()

    # Return the two filtered datasets
    return new_df_female, new_df_male
