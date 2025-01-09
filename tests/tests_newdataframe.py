import pandas as pd
import unittest 
import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.packages.new_data_frames import meaning_the_sessions
# from src.packages.new_data_frames import remove_outliers
# plt.switch_backend('Agg')
# Add the `src` directory to Python's search path
# Import the function to test
#from src.packages.new_data_frames import meaning_the_sessions as meaning_the_sessions
#first i want to test meaning the sessions
#positive test- i want to check that it handles simple data corecctly and means session in a simple data framev
def remove_outliers(data):
    """
    Replace outliers (values more than ±2.5 standard deviations from the mean) with NaN.
    """
    # Create a copy to avoid modifying the original data in place
    data = data.copy()

    for column in data.select_dtypes(include=[np.number]):  # Only process numeric columns
        mean = data[column].mean()
        std = data[column].std()

        # Handle cases where std is zero (no variance in data)
        if std == 0:
            continue

        lower_limit = mean - 2.5 * std
        upper_limit = mean + 2.5 * std

        # Replace outliers with NaN
        data[column] = data[column].apply(lambda x: np.nan if x < lower_limit or x > upper_limit else x)

    return data

    return data

def meaning_the_sessions(data):
    columns = data.columns
    identical_columns_dict = {}
    
    # Strip the last part of column names (e.g., '_session1', '_session2')
    stripped_columns = [re.sub(r'\d$', '', col) for col in columns if col != 'sex']
    unique_columns = set(stripped_columns)
    
    # Group columns based on their base name
    for column_uni in unique_columns:
        matching_columns = [col for col in columns if re.sub(r'\d$', '', col) == column_uni]
        identical_columns_dict[column_uni] = matching_columns
        
        # Exclude 'sex' column and empty matching columns
        matching_columns = [col for col in matching_columns if col != 'sex' and not data[col].isnull().all()]
        
        if not matching_columns:
            # Skip processing if no valid matching columns remain
            continue
        
        # Convert to numeric, ignoring errors, and skip non-numeric columns
        numeric_data = data[matching_columns].apply(pd.to_numeric, errors='coerce')
        
        if numeric_data.empty:
            # Skip processing if no numeric columns exist
            continue
        
        # Calculate means without filling NaN values
        row_means = numeric_data.mean(axis=1, skipna=True)
        data[column_uni + '_mean'] = row_means

    return data


class TestMeaningTheSessions(unittest.TestCase):

    def test_positive_case(self):
        # Positive case with valid data
        data = pd.DataFrame({
            'A_session1': [1, 2, 3],
            'A_session2': [4, 5, 6],
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        self.assertIn('A_session_mean', result.columns)
        self.assertListEqual(result['A_session_mean'].tolist(), [2.5, 3.5, 4.5])

    def test_negative_case(self):
    # Negative case with non-numeric values
        data = pd.DataFrame({
            'A_session1': ['1', 'two', '3'],  # 'two' is non-numeric
            'A_session2': [4, 5, 'six'],      # 'six' is non-numeric
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        
        # Check that 'A_session_mean' column exists
        self.assertIn('A_session_mean', result.columns)
        
        # Compute the expected result:
        # Row 1: mean of [1, 4] = (1+4)/2 = 2.5
        # Row 2: mean of [5] = 5 (only valid numeric value)
        # Row 3: mean of [3] = 3 (only valid numeric value)
        expected_means = [2.5, 5.0, 3.0]
        
        # Check if the result matches the expected means
        self.assertListEqual(result['A_session_mean'].tolist(), expected_means)

    def test_boundary_case(self):
        # Boundary case: Empty DataFrame
        data = pd.DataFrame(columns=['A_session1', 'A_session2', 'sex'])
        result = meaning_the_sessions(data)
        self.assertTrue(result.empty)

    def test_null_case(self):
        # Null case: DataFrame with NaN values
        data = pd.DataFrame({
            'A_session1': [1, None, 3],
            'A_session2': [4, 5, None],
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        self.assertIn('A_session_mean', result.columns)
        self.assertListEqual(result['A_session_mean'].tolist(), [2.5, 5.0, 3.0])

    def test_edge_case(self):
        # Edge case: Single-column DataFrame
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        self.assertIn('A_mean', result.columns)
        self.assertListEqual(result['A_mean'].tolist(), [1, 2, 3])



class TestRemoveOutliers(unittest.TestCase):

    def test_positive_case(self):
        """Test with normal data where outliers are replaced."""
        data = pd.DataFrame({
            'A': [10, 20, 30, 100000, 40, 50],
            'B': [1, 2, 3, 100, 4, 5]
        })
        result = remove_outliers(data.copy())
        self.assertTrue(np.isnan(result['A'][3]), "Outlier in column 'A' should be NaN")
        self.assertTrue(np.isnan(result['B'][3]), "Outlier in column 'B' should be NaN")
        self.assertEqual(result['A'][0], 10, "Non-outlier values should remain unchanged")

    def test_edge_case_empty_dataframe(self):
        """Test with an empty DataFrame."""
        data = pd.DataFrame()
        result = remove_outliers(data.copy())
        pd.testing.assert_frame_equal(result, data, "Empty DataFrame should remain unchanged")

    def test_edge_case_no_numeric_columns(self):
        """Test with a DataFrame that has no numeric columns."""
        data = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['x', 'y', 'z']
        })
        result = remove_outliers(data.copy())
        pd.testing.assert_frame_equal(result, data, "DataFrame with no numeric columns should remain unchanged")

    def test_boundary_case_within_limits(self):
        """Test with values on the boundary of being outliers."""
        data = pd.DataFrame({
            'A': [10, 20, 30, 40, 50],
        })
        result = remove_outliers(data.copy())
        self.assertFalse(result.isnull().values.any(), "No values should be NaN when within boundary")
   
    def test_boundary_case_exact_limits(self):
        """Test with values exactly at the boundary."""
        data = pd.DataFrame({
            'A': [10, 20, 30, 40, 50]
        })
        mean = data['A'].mean()
        std = data['A'].std()

        # Round to handle floating-point precision issues
        lower_bound = round(mean - 2.5 * std, 6)
        upper_bound = round(mean + 2.5 * std, 6)

        # Add the boundary values as separate rows using pd.concat
        boundary_data = pd.DataFrame({'A': [lower_bound, upper_bound]})
        data = pd.concat([data, boundary_data], ignore_index=True)

        # Run the function
        result = remove_outliers(data.copy())

        # Assert no boundary values are replaced with NaN
        self.assertFalse(result.isnull().values.any(), "Boundary values should not be considered outliers")


if __name__ == "__main__":
    unittest.main()




    