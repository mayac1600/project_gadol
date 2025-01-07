import pandas as pd
import pytest
import unittest 
import re
import sys
import os

# Add the `src` directory to Python's search path
# Import the function to test
#from src.packages.new_data_frames import meaning_the_sessions as meaning_the_sessions
#first i want to test meaning the sessions
#positive test- i want to check that it handles simple data corecctly and means session in a simple data frame
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
        # Exclude 'sex' column from numeric processing
        if 'sex' in matching_columns:
            matching_columns.remove('sex')
        numeric_data = data[matching_columns].apply(pd.to_numeric, errors='coerce')
        if not numeric_data.empty:
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
        self.assertIn('A_mean', result.columns)
        self.assertListEqual(result['A_mean'].tolist(), [2.5, 3.5, 4.5])

    def test_negative_case(self):
        # Negative case with non-numeric values
        data = pd.DataFrame({
            'A_session1': ['1', 'two', '3'],
            'A_session2': [4, 5, 'six'],
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        self.assertIn('A_mean', result.columns)
        self.assertListEqual(result['A_mean'].tolist(), [2.5, 5.0, 3.0])

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
        self.assertIn('A_mean', result.columns)
        self.assertListEqual(result['A_mean'].tolist(), [2.5, 5.0, 3.0])

    def test_edge_case(self):
        # Edge case: Single-column DataFrame
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'sex': ['M', 'F', 'M']
        })
        result = meaning_the_sessions(data)
        self.assertIn('A_mean', result.columns)
        self.assertListEqual(result['A_mean'].tolist(), [1, 2, 3])
if __name__ == '__main__':
    unittest.main()