import unittest 
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from unittest.mock import patch
from statsmodels.regression.linear_model import RegressionResultsWrapper
from packages.tools.anlysis import linear_regression_trial as linear_regression_trial
from packages.tools.anlysis import plot_signi as plot_signi
from packages.tools.anlysis import linear_regression_with_sex_interactions as linear_regression_with_sex_interactions
from packages.tools.anlysis import plot_signi_bysex as plot_signi_bysex

logging.basicConfig(level=logging.INFO)

class TestLinearRegressionTrial(unittest.TestCase):
    def test_identify_relevant_columns(self):
        data = pd.DataFrame({
            "col1_mean": [1, 2, 3],
            "col2_mean": [4, 5, 6],
            "irrelevant_col": [7, 8, 9]
        })

        expected_columns = ["col1_mean", "col2_mean"]
        actual_columns = [col for col in data.columns if "_mean" in col]

        self.assertEqual(actual_columns, expected_columns)

    def test_fill_missing_values(self):
        data = pd.DataFrame({
            "col1_mean": [1, None, 3],
            "col2_mean": [None, 5, 6],
        })

        expected_data = pd.DataFrame({
            "col1_mean": [1.0, 2.0, 3.0],
            "col2_mean": [5.5, 5.0, 6.0],
        })

        data[["col1_mean", "col2_mean"]] = data[["col1_mean", "col2_mean"]].apply(
            lambda col: col.fillna(col.mean()), axis=0
        )

        pd.testing.assert_frame_equal(data, expected_data)

    def test_separate_predictors_and_response(self):
        data = pd.DataFrame({
            "Predictor1_mean": [1, 2, 3],
            "Predictor2_mean": [4, 5, 6],
            "Mini_Item1_mean": [7, 8, 9],
            "Mini_Item2_mean": [10, 11, 12]
        })

        expected_predictors = data[["Predictor1_mean", "Predictor2_mean"]]
        expected_response = ["Mini_Item1_mean", "Mini_Item2_mean"]

        predictors = data[[col for col in data.columns if "Mini_Item" not in col]]
        response = [col for col in data.columns if "Mini_Item" in col]

        pd.testing.assert_frame_equal(predictors, expected_predictors)
        self.assertEqual(response, expected_response)

    @patch("matplotlib.pyplot.show")
    def test_plot_generation(self, mock_show):
        data = pd.DataFrame({
            "Predictor1_mean": [1, 2, 3],
            "Mini_Item12_EO_mean": [4, 5, 6]
        })

        linear_regression_trial(data)

        fig = plt.gcf()
        axes = fig.get_axes()

        # Check the title for the transformed name
        self.assertEqual(axes[0].get_title(), "P-values for EO_mean")

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")  # Patch plt.show to suppress plot display
    def test_linear_regression_trial_functional(self, mock_show):
        """
        Test the linear_regression_trial function to ensure it produces valid regression models.
        """
        data = pd.DataFrame({
            "Predictor1_mean": [1, None, 3],
            "Predictor2_mean": [None, 5, 6],
            "Mini_Item1_mean": [7, 8, 9],
            "Mini_Item2_mean": [10, 11, None]
        })

        models = linear_regression_trial(data)

        # Check that the models dictionary is not empty
        self.assertTrue(models, "The models dictionary is empty.")

        # Ensure all items in the models dictionary are RegressionResultsWrapper instances
        self.assertTrue(
            all(isinstance(models[item], RegressionResultsWrapper) for item in models),
            "Not all items in the models dictionary are RegressionResultsWrapper instances."
        )

class TestPlotSigni(unittest.TestCase):
    """
    Testing class for the `plot_signi` function. Ensures proper handling of
    significant predictors, missing variables, empty data, and regression plotting.
    """

    @patch("matplotlib.pyplot.show")
    def test_plot_significant_predictors(self, mock_show):
        """
        Test that only significant predictors (p-value <= 0.05) are plotted.
        Validates plot content, axes labels, and legends for correctness.
        """
        mock_model = unittest.mock.Mock(spec=RegressionResultsWrapper)
        mock_model.pvalues = pd.Series({
            "Predictor1": 0.03,  # Significant predictor
            "Predictor2": 0.06,  # Non-significant predictor
            "const": 0.01        # Intercept (ignored)
        })
        mock_models = {"ResponseVar": mock_model}

        data = pd.DataFrame({
            "Predictor1": [1, 2, 3, 4, 5],
            "Predictor2": [5, 4, 3, 2, 1],  # Should not be plotted
            "ResponseVar": [2, 4, 6, 8, 10]
        })

        with patch("matplotlib.pyplot.figure") as mock_figure:
            plot_signi(mock_models, data)
            self.assertEqual(mock_figure.call_count, 1)

        fig = plt.gcf()
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_xlabel(), "Predictor1")
        self.assertEqual(axes[0].get_ylabel(), "ResponseVar")

        legend_texts = [text.get_text() for text in axes[0].get_legend().texts]
        self.assertTrue(any("Regression Line" in text for text in legend_texts))
        mock_show.assert_called()

    @patch("matplotlib.pyplot.show")  # Patch plt.show to suppress plot display
    def test_missing_state_variable(self, mock_show):
        """
        Test that the function logs a message when a state variable is missing from the data.
        """
        # Mock RegressionResultsWrapper
        mock_model = unittest.mock.Mock(spec=RegressionResultsWrapper)
        mock_model.pvalues = pd.Series({"Predictor1": 0.03, "const": 0.01})

        # Mock model dictionary with a missing state variable
        mock_models = {"MissingVar": mock_model}  # State variable is missing in data

        # Sample data without 'MissingVar'
        data = pd.DataFrame({"Predictor1": [1, 2, 3], "ResponseVar": [4, 5, 6]})

        # Capture logs
        with self.assertLogs(level="INFO") as log:
            plot_signi(mock_models, data)

        # Check that the log message contains the expected variable name
        self.assertTrue(
            any("MissingVar" in message for message in log.output),
            "Expected variable name 'MissingVar' not found in logs."
        )


    @patch("matplotlib.pyplot.show")
    def test_no_valid_data(self, mock_show):
        """
        Test that the function logs a message when all data is NaN for a predictor and state variable.
        """
        mock_model = unittest.mock.Mock(spec=RegressionResultsWrapper)
        mock_model.pvalues = pd.Series({"Predictor1": 0.03, "const": 0.01})
        mock_models = {"ResponseVar": mock_model}  # Valid state variable

        data = pd.DataFrame({"Predictor1": [np.nan, np.nan], "ResponseVar": [np.nan, np.nan]})

        with self.assertLogs(level="INFO") as log:
            plot_signi(mock_models, data)

        # Check that the log message contains the expected variable and predictor names
        self.assertTrue(
            any("Predictor1" in message and "ResponseVar" in message for message in log.output),
            "Expected 'Predictor1' or 'ResponseVar' not found in logs."
        )
    @patch("matplotlib.pyplot.show")
    def test_plot_regression_line(self, mock_show):
        """
        Test that the function correctly plots a regression line,
        with proper labels, legends, and R^2 values.
        """
        mock_model = unittest.mock.Mock(spec=RegressionResultsWrapper)
        mock_model.pvalues = pd.Series({"Predictor1": 0.03, "const": 0.01})
        mock_models = {"ResponseVar": mock_model}

        data = pd.DataFrame({"Predictor1": [1, 2, 3], "ResponseVar": [2, 4, 6]})

        plot_signi(mock_models, data)

        fig = plt.gcf()
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_xlabel(), "Predictor1")
        self.assertEqual(axes[0].get_ylabel(), "ResponseVar")

        legend_texts = [text.get_text() for text in axes[0].get_legend().texts]
        self.assertTrue(any("Regression Line" in text for text in legend_texts))
        mock_show.assert_called()

class TestLinearRegressionWithSexInteractions(unittest.TestCase):
    """
    Test class for `linear_regression_with_sex_interactions` and `plot_signi_bysex`.
    """

    @patch("matplotlib.pyplot.show")
    def test_interaction_terms_generated(self, mock_show):
        """
        Test that interaction terms (_bysex) are correctly generated in the output DataFrame.
        """
        data = pd.DataFrame({
            "Predictor1_mean": [1, 2, 3, np.nan],
            "Predictor2_mean": [4, np.nan, 6, 8],
            "Mini_Item1_mean": [10, 20, 30, 40],
            "sex": ["m", "f", "m", "f"]
        })

        _, processed_data = linear_regression_with_sex_interactions(data)

        # Verify interaction terms exist
        self.assertIn("Predictor1_mean_bysex", processed_data.columns)
        self.assertIn("Predictor2_mean_bysex", processed_data.columns)

        # Verify interaction values are correctly computed
        self.assertEqual(processed_data["Predictor1_mean_bysex"].iloc[1], 2)  # Predictor1 * sex (1 for female)
        self.assertEqual(processed_data["Predictor1_mean_bysex"].iloc[0], 0)  # Predictor1 * sex (0 for male)

    @patch("matplotlib.pyplot.show")
    def test_missing_sex_column(self, mock_show):
        """
        Test that the function handles missing 'sex' column gracefully.
        """
        data = pd.DataFrame({
            "Predictor1_mean": [1, 2, 3, 4],
            "Mini_Item1_mean": [10, 20, 30, 40]
        })

        with self.assertRaises(KeyError):
            linear_regression_with_sex_interactions(data)

    @patch("matplotlib.pyplot.show")
    def test_no_valid_data_for_plot(self, mock_show):
        """
        Test that the function skips plotting when no valid data exists for male or female.
        """
        models = {"Mini_Item1_mean": unittest.mock.Mock(spec=RegressionResultsWrapper)}
        models["Mini_Item1_mean"].pvalues = pd.Series({"Predictor1_mean_bysex": 0.03, "const": 0.01})

        maledata = pd.DataFrame({
            "Predictor1_mean": [np.nan, np.nan],
            "Mini_Item1_mean": [np.nan, np.nan]
        })
        femaledata = pd.DataFrame({
            "Predictor1_mean": [np.nan, np.nan],
            "Mini_Item1_mean": [np.nan, np.nan]
        })

        with self.assertLogs(level="INFO") as log:
            plot_signi_bysex(models, None, maledata, femaledata)

        self.assertTrue(any("No valid data for Predictor1_mean" in message for message in log.output))
    @patch("matplotlib.pyplot.show")
    def test_no_valid_data_for_plot(self, mock_show):
        """
        Test that the function skips plotting when no valid data exists for male or female.
        """
        models = {"Mini_Item1_mean": unittest.mock.Mock(spec=RegressionResultsWrapper)}
        models["Mini_Item1_mean"].pvalues = pd.Series({"Predictor1_mean_bysex": 0.03, "const": 0.01})

        maledata = pd.DataFrame({
            "Predictor1_mean": [np.nan, np.nan],
            "Mini_Item1_mean": [np.nan, np.nan]
        })
        femaledata = pd.DataFrame({
            "Predictor1_mean": [np.nan, np.nan],
            "Mini_Item1_mean": [np.nan, np.nan]
        })

        with self.assertLogs(level="INFO") as log:
            plot_signi_bysex(models, None, maledata, femaledata)

        self.assertTrue(
            any("No valid data for Predictor1_mean" in message for message in log.output),
            "Expected log message for missing valid data not found."
        )

if __name__ == "__main__":
    unittest.main()


