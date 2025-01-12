import pandas as pd
import unittest 
import numpy as np
from packages.tools.new_data_frames import remove_outliers
from packages.tools.new_data_frames import meaning_the_sessions
from packages.tools.new_data_frames import separating_genders

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

class TestSeparatingGenders(unittest.TestCase):

    def test_normal_data(self):
        """Test with normal data containing both males and females."""
        data = pd.DataFrame({
            'sex': ['m', 'f', 'm', 'f'],
            'age': [25, 30, 35, 40]
        })
        new_df_female, new_df_male = separating_genders(data)
        
        # Assert females
        self.assertEqual(len(new_df_female), 2)
        self.assertListEqual(new_df_female['sex'].tolist(), ['f', 'f'])
        self.assertListEqual(new_df_female['age'].tolist(), [30, 40])
        
        # Assert males
        self.assertEqual(len(new_df_male), 2)
        self.assertListEqual(new_df_male['sex'].tolist(), ['m', 'm'])
        self.assertListEqual(new_df_male['age'].tolist(), [25, 35])

    def test_missing_one_gender(self):
        """Test with data missing one gender."""
        data = pd.DataFrame({
            'sex': ['m', 'm', 'm'],
            'age': [25, 35, 45]
        })
        new_df_female, new_df_male = separating_genders(data)
        
        # Assert females
        self.assertEqual(len(new_df_female), 0)  # Should be empty
        self.assertTrue(new_df_female.empty)

        # Assert males
        self.assertEqual(len(new_df_male), 3)
        self.assertListEqual(new_df_male['sex'].tolist(), ['m', 'm', 'm'])
        self.assertListEqual(new_df_male['age'].tolist(), [25, 35, 45])

    def test_ignore_invalid_values(self):
        """Test with invalid values in the 'sex' column."""
        data = pd.DataFrame({
            'sex': ['m', 'f', 'unknown', 'other', 'f', 'M', 'F '],
            'age': [25, 30, 35, 40, 45, 50, 55]
        })
        new_df_female, new_df_male = separating_genders(data)
        
        # Assert females
        self.assertEqual(len(new_df_female), 3)
        self.assertListEqual(new_df_female['sex'].tolist(), ['f', 'f', 'f'])
        self.assertListEqual(new_df_female['age'].tolist(), [30, 45, 55])
        
        # Assert males
        self.assertEqual(len(new_df_male), 2)
        self.assertListEqual(new_df_male['sex'].tolist(), ['m', 'm'])
        self.assertListEqual(new_df_male['age'].tolist(), [25, 50])

    def test_missing_sex_column(self):
        """Test when the 'sex' column is missing."""
        data = pd.DataFrame({
            'age': [25, 30, 35]
        })
        with self.assertRaises(KeyError):
            separating_genders(data)

    def test_case_insensitivity_and_whitespace(self):
        """Test case insensitivity and handling of extra whitespace."""
        data = pd.DataFrame({
            'sex': [' M ', ' f ', ' M', 'F'],
            'age': [25, 30, 35, 40]
        })
        new_df_female, new_df_male = separating_genders(data)
        
        # Assert females
        self.assertEqual(len(new_df_female), 2)
        self.assertListEqual(new_df_female['sex'].tolist(), ['f', 'f'])
        self.assertListEqual(new_df_female['age'].tolist(), [30, 40])
        
        # Assert males
        self.assertEqual(len(new_df_male), 2)
        self.assertListEqual(new_df_male['sex'].tolist(), ['m', 'm'])
        self.assertListEqual(new_df_male['age'].tolist(), [25, 35])


if __name__ == "__main__":
    unittest.main()




    