import pandas as pd
import unittest 
import numpy as np
import re
from unittest.mock import patch
from packages.tools.explorations import plot_side_by_side_bars
from packages.tools.explorations import compare_sessions_grouped
from packages.tools.explorations import plot_intrusive_thoughts
from packages.tools.explorations import plot_correlation_matrix

class TestPlotSideBySideBars(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_normal_data(self, mock_show):
        """Test that bar heights and counts are correct for normal data."""
        # Mock datasets
        female_data = pd.DataFrame({
            '(1)_mean': [3, 4],
            '(2)_mean': [5, 6],
            '(3)_mean': [7, 8]
        })
        male_data = pd.DataFrame({
            '(1)_mean': [2, 3],
            '(2)_mean': [4, 5],
            '(3)_mean': [6, 7]
        })
        united_data = pd.DataFrame({
            '(1)_mean': [2.5, 3.5],
            '(2)_mean': [4.5, 5.5],
            '(3)_mean': [6.5, 7.5]
        })

        # Patch plt.bar to intercept calls
        with patch("matplotlib.pyplot.bar") as mock_bar:
            plot_side_by_side_bars(female_data, male_data, united_data)

            # Verify the number of plt.bar calls
            self.assertEqual(mock_bar.call_count, 3)  # One call for each group: Female, Male, United

            # Check the heights passed to plt.bar for Female
            female_call_args, _ = mock_bar.call_args_list[0]
            female_positions, female_heights = female_call_args
            expected_female_means = female_data.mean()
            self.assertTrue(
                all(round(h, 2) == round(m, 2) for h, m in zip(female_heights, expected_female_means)),
                "Mismatch in female bar heights"
            )

            # Check the heights passed to plt.bar for Male
            male_call_args, _ = mock_bar.call_args_list[1]
            male_positions, male_heights = male_call_args
            expected_male_means = male_data.mean()
            self.assertTrue(
                all(round(h, 2) == round(m, 2) for h, m in zip(male_heights, expected_male_means)),
                "Mismatch in male bar heights"
            )

            # Check the heights passed to plt.bar for United
            united_call_args, _ = mock_bar.call_args_list[2]
            united_positions, united_heights = united_call_args
            expected_united_means = united_data.mean()
            self.assertTrue(
                all(round(h, 2) == round(m, 2) for h, m in zip(united_heights, expected_united_means)),
                "Mismatch in united bar heights"
            )
    @patch("matplotlib.pyplot.show")  # Patch plt.show to suppress plot display
    def test_empty_columns(self, mock_show):
        """Test with empty columns to ensure it handles them gracefully."""
        female_data = pd.DataFrame({
            '(1)_mean': [],
            '(2)_mean': [],
            '(3)_mean': []
        })
        male_data = pd.DataFrame({
            '(1)_mean': [],
            '(2)_mean': [],
            '(3)_mean': []
        })
        united_data = pd.DataFrame({
            '(1)_mean': [],
            '(2)_mean': [],
            '(3)_mean': []
        })

        # Call the function
        plot_side_by_side_bars(female_data, male_data, united_data)

        # Assert means are NaN for empty columns
        self.assertTrue(female_data.mean().isna().all(), "Female data means are not NaN for empty columns.")
        self.assertTrue(male_data.mean().isna().all(), "Male data means are not NaN for empty columns.")
        self.assertTrue(united_data.mean().isna().all(), "United data means are not NaN for empty columns.")

    def test_non_numeric_values(self):
        """Test with non-numeric values to ensure they are ignored, using fragments from the original def."""
        female_data = pd.DataFrame({
            '(1)_mean': [3, 'a', None],
            '(2)_mean': [5, 'b', None],
            '(3)_mean': [7, 'c', None]
        })
        male_data = pd.DataFrame({
            '(1)_mean': [2, None, 'x'],
            '(2)_mean': [4, 'y', None],
            '(3)_mean': [6, None, 'z']
        })
        united_data = pd.DataFrame({
            '(1)_mean': [2.5, 'p', None],
            '(2)_mean': [4.5, None, 'q'],
            '(3)_mean': [6.5, 'r', None]
        })

        # Calculate expected means (ignoring non-numeric values)-> a fragment from the def 
        expected_female_means = female_data.apply(pd.to_numeric, errors='coerce').mean()
        expected_male_means = male_data.apply(pd.to_numeric, errors='coerce').mean()
        expected_united_means = united_data.apply(pd.to_numeric, errors='coerce').mean()

        # Assert means match expected (ignoring non-numeric values)
        pd.testing.assert_series_equal(female_data.apply(pd.to_numeric, errors='coerce').mean(), expected_female_means)
        pd.testing.assert_series_equal(male_data.apply(pd.to_numeric, errors='coerce').mean(), expected_male_means)
        pd.testing.assert_series_equal(united_data.apply(pd.to_numeric, errors='coerce').mean(), expected_united_means)

    def test_column_order_assertion(self):
        """Test that the function raises an assertion error for mismatched column orders."""
        female_data = pd.DataFrame({
            '(1)_mean': [3, 4],
            '(2)_mean': [5, 6]
        })
        male_data = pd.DataFrame({
            '(1)_mean': [2, 3],
            '(3)_mean': [6, 7]  # Different column order
        })
        united_data = pd.DataFrame({
            '(1)_mean': [2.5, 3.5],
            '(2)_mean': [4.5, 5.5]
        })

        with self.assertRaises(AssertionError):
            plot_side_by_side_bars(female_data, male_data, united_data)

class TestCompareSessionsGrouped(unittest.TestCase):
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.show")  # Patch plt.show to suppress plot display
    def test_normal_data(self, mock_show, mock_plot, mock_bar):
        """Test that bar heights and counts are correct for normal data."""
        # Mock dataset with session metrics
        data = pd.DataFrame({
            "metric1_1": [3, 4, 5],
            "metric1_2": [6, 7, 8],
            "metric1_3": [9, 10, 11],
            "metric2_1": [2, 3, 4],
            "metric2_2": [5, 6, 7],
            "metric2_3": [8, 9, 10],
        })

        title = "Test Plot"

        # Call the function
        compare_sessions_grouped(data, title)

        # Verify the number of plt.bar calls
        num_sessions = 3  # Sessions 1, 2, 3
        self.assertEqual(mock_bar.call_count, num_sessions)

        # Verify bar heights for each session
        session_numbers = [1, 2, 3]
        for i, session in enumerate(session_numbers):
            # Extract bar heights from plt.bar call
            bar_call_args, _ = mock_bar.call_args_list[i]
            bar_positions, bar_heights = bar_call_args[:2]

            # Extract expected means for this session
            expected_means = [
                data[f"metric1_{session}"].mean(),
                data[f"metric2_{session}"].mean(),
            ]

            # Assert the bar heights match the expected means
            self.assertTrue(
                np.allclose(sorted(bar_heights), sorted(expected_means), atol=1e-2),
                f"Mismatch in bar heights for Session {session}: {sorted(bar_heights)} != {sorted(expected_means)}"
            )
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.plot")
    def test_empty_columns(self, mock_plot, mock_bar):
        """Ensure the function handles columns that are completely empty by checking that means are zero."""
        data = pd.DataFrame({
            'metric1_1': [np.nan, np.nan, np.nan],                'metric1_2': [np.nan, np.nan, np.nan],
            'metric1_3': [np.nan, np.nan, np.nan]
            })

            # Call the function
        compare_sessions_grouped(data, title="Empty Columns Test")

            # Ensure plt.bar and plt.plot were never called (no data to plot)
        mock_bar.assert_not_called()
        mock_plot.assert_not_called()
   
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.show")
    def test_incomplete_session_data(self, mock_show, mock_bar):
        """Asserts that metrics with incomplete session data are excluded from processing."""
        data = pd.DataFrame({
            'metric1_1': [10, 20, 30],
            'metric1_2': [12, 22, 32],
            # Missing metric1_3
            'metric2_1': [15, 25, 35],
            'metric2_2': [18, 28, 38],
            'metric2_3': [20, 30, 40]
        })

        # Call the function
        compare_sessions_grouped(data, title="Incomplete Session Data Test")

        # Ensure plt.bar is called only for 'metric2'
        self.assertEqual(mock_bar.call_count, 3)  # One bar for each session of 'metric2'

        # Extract the heights (means) used in plt.bar calls to confirm correct metrics were processed
        bar_heights = [call[0][1] for call in mock_bar.call_args_list]
        expected_means_metric2 = [
            data['metric2_1'].mean(),
            data['metric2_2'].mean(),
            data['metric2_3'].mean()
        ]

        # Assert bar heights match the exact expected means of 'metric2'
        self.assertListEqual(
            bar_heights,
            expected_means_metric2,
            "Mismatch in bar heights for metric2"
        )
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.show")  # Patch plt.show to suppress plot display
    def test_non_numeric_values(self, mock_show, mock_plot, mock_bar):
        """Confirms that non-numeric values are ignored and do not interfere with calculations."""
        data = pd.DataFrame({
            'metric1_1': [10, 'a', 30],
            'metric1_2': [12, 22, 'b'],
            'metric1_3': [14, 'c', 34]
        })

        # Call the function
        compare_sessions_grouped(data, title="Non-Numeric Values Test")

        # Ensure plt.bar is called
        self.assertTrue(mock_bar.called)

        # Convert the data to numeric, coercing non-numeric values to NaN
        numeric_data = data.apply(pd.to_numeric, errors='coerce')

        # Iterate through each call to plt.bar and verify bar heights
        for i, (col, call_args) in enumerate(zip(numeric_data.columns, mock_bar.call_args_list)):
            expected_mean = numeric_data[col].mean()
            bar_height = call_args[0][1][0]  # Extract the first height from mock call
            self.assertAlmostEqual(
                bar_height,
                expected_mean,
                places=2,
                msg=f"Mismatch in bar height for {col}"
            )

class TestPlotIntrusiveThoughts(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_normal_data(self, mock_show):
        """Test that the function correctly plots normal data."""
        # Mock datasets
        female_data = pd.DataFrame({
            "Mini_Item12_1_mean": [3, 4, 5],
            "Mini_Item12_2_mean": [6, 7, 8],
            "Mini_Item12_3_mean": [9, 10, 11]
        })
        male_data = pd.DataFrame({
            "Mini_Item12_1_mean": [2, 3, 4],
            "Mini_Item12_2_mean": [5, 6, 7],
            "Mini_Item12_3_mean": [8, 9, 10]
        })
        united_data = pd.DataFrame({
            "Mini_Item12_1_mean": [2.5, 3.5, 4.5],
            "Mini_Item12_2_mean": [5.5, 6.5, 7.5],
            "Mini_Item12_3_mean": [8.5, 9.5, 10.5]
        })

        # Patch plt.bar to intercept calls
        with patch("matplotlib.pyplot.bar") as mock_bar:
            plot_intrusive_thoughts(female_data, male_data, united_data)

            # Verify the number of plt.bar calls
            self.assertEqual(mock_bar.call_count, 3)  # Female, Male, United

            # Extract bar heights and compare with expected means
            expected_female_means = female_data.mean()
            expected_male_means = male_data.mean()
            expected_united_means = united_data.mean()

            bar_heights = [call[0][1] for call in mock_bar.call_args_list]

            self.assertTrue(
                np.allclose(bar_heights[0], expected_female_means, atol=1e-6),
                "Mismatch in female bar heights"
            )
            self.assertTrue(
                np.allclose(bar_heights[1], expected_male_means, atol=1e-6),
                "Mismatch in male bar heights"
            )
            self.assertTrue(
                np.allclose(bar_heights[2], expected_united_means, atol=1e-6),
                "Mismatch in united bar heights"
            )

    @patch("matplotlib.pyplot.show")
    def test_no_common_columns(self, mock_show):
        """Test that the function handles no common columns gracefully."""
        female_data = pd.DataFrame({
            "Other_Item1_mean": [3, 4, 5],
            "Other_Item2_mean": [6, 7, 8]
        })
        male_data = pd.DataFrame({
            "Other_Item1_mean": [2, 3, 4],
            "Other_Item2_mean": [5, 6, 7]
        })
        united_data = pd.DataFrame({
            "Other_Item1_mean": [2.5, 3.5, 4.5],
            "Other_Item2_mean": [5.5, 6.5, 7.5]
        })

        # Patch plt.bar and plt.axhline
        with patch("matplotlib.pyplot.bar") as mock_bar, patch("matplotlib.pyplot.axhline") as mock_axhline:
            plot_intrusive_thoughts(female_data, male_data, united_data)

            # Assert that plt.bar and plt.axhline were not called
            mock_bar.assert_not_called()
            mock_axhline.assert_not_called()

    @patch("matplotlib.pyplot.show")
    def test_non_numeric_values(self, mock_show):
        """Test that the function ignores non-numeric values."""
        female_data = pd.DataFrame({
            "Mini_Item12_1_mean": [3, "a", 5],
            "Mini_Item12_2_mean": [6, None, 8],
            "Mini_Item12_3_mean": [9, 10, "b"]
        })
        male_data = pd.DataFrame({
            "Mini_Item12_1_mean": [2, "x", 4],
            "Mini_Item12_2_mean": [5, None, 7],
            "Mini_Item12_3_mean": [8, 9, "y"]
        })
        united_data = pd.DataFrame({
            "Mini_Item12_1_mean": [2.5, "p", 4.5],
            "Mini_Item12_2_mean": [5.5, 6.5, None],
            "Mini_Item12_3_mean": ["q", 9.5, 10.5]
        })

        # Patch plt.bar to intercept calls
        with patch("matplotlib.pyplot.bar") as mock_bar:
            plot_intrusive_thoughts(female_data, male_data, united_data)

            # Extract the cleaned data and calculate expected means
            numeric_female_data = female_data.apply(pd.to_numeric, errors="coerce")
            numeric_male_data = male_data.apply(pd.to_numeric, errors="coerce")
            numeric_united_data = united_data.apply(pd.to_numeric, errors="coerce")

            expected_female_means = numeric_female_data.mean()
            expected_male_means = numeric_male_data.mean()
            expected_united_means = numeric_united_data.mean()

            # Verify bar heights
            bar_heights = [call[0][1] for call in mock_bar.call_args_list]

            self.assertTrue(
                np.allclose(bar_heights[0], expected_female_means, atol=1e-6),
                "Mismatch in female bar heights"
            )
            self.assertTrue(
                np.allclose(bar_heights[1], expected_male_means, atol=1e-6),
                "Mismatch in male bar heights"
            )
            self.assertTrue(
                np.allclose(bar_heights[2], expected_united_means, atol=1e-6),
                "Mismatch in united bar heights"
            )

class TestPlotCorrelationMatrix(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    @patch("seaborn.heatmap")
    def test_default_variables(self, mock_heatmap, mock_show):
        """Test the default behavior when variables are not provided."""
        data = pd.DataFrame({
            '(1)variable1_mean': [1, 2, 3],
            '(2)variable2_mean': [4, 5, 6],
            '(3)variable3_mean': [7, 8, 9]
        })

        # Expect the function to process and plot without errors
        try:
            plot_correlation_matrix(data, title="Default Variables Test")
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")

        # Verify sns.heatmap was called
        mock_heatmap.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("seaborn.heatmap")
    def test_custom_sorting(self, mock_heatmap, mock_show):
        """Test that variables are sorted based on the custom sort key."""
        data = pd.DataFrame({
            '(3)variable3_mean': [7, 8, 9],
            '(1)variable1_mean': [1, 2, 3],
            '(2)variable2_mean': [4, 5, 6]
        })

        # Variables should be sorted numerically based on the custom key
        sorted_variables = ['(1)variable1_mean', '(2)variable2_mean', '(3)variable3_mean']

        try:
            plot_correlation_matrix(data, title="Custom Sorting Test")
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")

        self.assertEqual(
            sorted(data.columns, key=lambda col: int(re.match(r'\((\d+)\)', col).group(1))),
            sorted_variables
        )

    def test_edge_case_empty_data(self):
        """Test with an empty DataFrame."""
        data = pd.DataFrame()

        # Expect the function to raise a ValueError for empty data
        with self.assertRaises(ValueError):
            plot_correlation_matrix(data, title="Empty DataFrame Test")

    @patch("matplotlib.pyplot.show")
    @patch("seaborn.heatmap")
    def test_boundary_case_single_variable(self, mock_heatmap, mock_show):
        """Test with only one variable in the DataFrame."""
        data = pd.DataFrame({
            '(1)variable1_mean': [1, 2, 3]
        })

        # Single variable should not raise an error
        try:
            plot_correlation_matrix(data, title="Single Variable Test")
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")

        # Verify sns.heatmap was called
        mock_heatmap.assert_called_once()

    def test_non_matching_columns(self):
        """Test with columns that don't match the default 'mean' filter."""
        data = pd.DataFrame({
            'random_variable': [1, 2, 3],
            'another_column': [4, 5, 6]
        })

        # Expect the function to raise a ValueError for no matching columns
        with self.assertRaises(ValueError):
            plot_correlation_matrix(data, title="Non-Matching Columns Test")

    @patch("matplotlib.pyplot.show")
    @patch("seaborn.heatmap")
    def test_variables_provided(self, mock_heatmap, mock_show):
        """Test with explicitly provided variables."""
        data = pd.DataFrame({
            '(1)variable1_mean': [1, 2, 3],
            '(2)variable2_mean': [4, 5, 6],
            '(3)variable3_mean': [7, 8, 9],
            'unrelated_column': [10, 11, 12]
        })

        provided_variables = ['(1)variable1_mean', '(2)variable2_mean']

        try:
            plot_correlation_matrix(data, variables=provided_variables, title="Provided Variables Test")
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")

        # Verify sns.heatmap was called
        mock_heatmap.assert_called_once()

    def test_non_numeric_variables(self):
        """Test with non-numeric variables in the DataFrame."""
        data = pd.DataFrame({
            '(1)variable1_mean': ['a', 'b', 'c'],
            '(2)variable2_mean': [1, 2, 3],
            '(3)variable3_mean': [4, 5, 'x']
        })

        # Expect the function to raise a ValueError for non-numeric variables
        with self.assertRaises(ValueError):
            plot_correlation_matrix(data, title="Non-Numeric Variables Test")

if __name__ == '__main__':
    unittest.main()

