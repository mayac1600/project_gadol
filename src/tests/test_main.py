import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('/Users/mayacohen/Desktop/project_gadol/data/participants.the.one.that.works.csv')

# List of desired columns
columns = [
    '(1)Discontinuity of Mind_session1', '(2)Theory of Mind_session1', '(3)Self_session1', '(4)Planning_session1',
    '(5)Sleepiness_session1', '(6)Comfort_session1', '(7)Somatic Awareness_session1', '(8)Health Concern_session1',
    '(9)Visual Thought_session1', '(10)Verbal Thought_session1', '(1)Discontinuity of Mind_session2',
    '(2)Theory of Mind_session2', '(3)Self_session2', '(4)Planning_session2', '(5)Sleepiness_session2',
    '(6)Comfort_session2', '(7)Somatic Awareness_session2', '(8)Health Concern_session2', '(9)Visual Thought_session2',
    '(10)Verbal Thought_session2', '(1)Discontinuity of Mind_session3', '(2)Theory of Mind_session3',
    '(3)Self_session3', '(4)Planning_session3', '(5)Sleepiness_session3', '(6)Comfort_session3',
    '(7)Somatic Awareness_session3', '(8)Health Concern_session3', '(9)Visual Thought_session3',
    '(10)Verbal Thought_session3',
    "Mini_Item12_EO1", "Mini_Item12_EC1", "Mini_Item12_Music1", "Mini_Item12_Memory1", "Mini_Item12_Subtraction1",
    "Mini_Item12_EO2", "Mini_Item12_EC2", "Mini_Item12_Music2", "Mini_Item12_Memory2", "Mini_Item12_Subtraction2",
    "Mini_Item12_EO3", "Mini_Item12_EC3", "Mini_Item12_Music3", "Mini_Item12_Memory3", "Mini_Item12_Subtraction3",'sex',
]
# Select the columns
new_df_mixed_genders = df[columns].copy()


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

new_df_mixed_genders = meaning_the_sessions(new_df_mixed_genders)



def separating_genders(data):
    # Ensure the 'sex' column exists
    if 'sex' not in data.columns:
        raise KeyError("'sex' column is missing in the DataFrame.")
    
    # Clean the 'sex' column to handle inconsistencies
    data['sex'] = data['sex'].astype(str).str.strip().str.lower()
    # Filter rows for males and females
    new_df_male = data[data['sex'] == 'm'].copy()
    new_df_female = data[data['sex'] == 'f'].copy()


    return new_df_female, new_df_male
# Call the function
new_df_female, new_df_male = separating_genders(new_df_mixed_genders)

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re

def linear_regression_trial(data):
    # Identify all relevant columns containing "mean"
    all_relevant_cols = [col for col in data.columns if "_mean" in col]
    print("all_relevant_cols:", all_relevant_cols)
    data[all_relevant_cols] = data[all_relevant_cols].apply(lambda col: col.fillna(col.mean()), axis=0)

    # Separate predictors and response variables
    mind_wandering_predictors = data[[col for col in all_relevant_cols if "Mini_Item" not in col]]
    intrusive_thoughts_predicted = [col for col in all_relevant_cols if "Mini_Item" in col]
    print("mind_wandering_predictors:\n", mind_wandering_predictors)
    print("intrusive_thoughts_predicted:\n", intrusive_thoughts_predicted)

    # Set up a grid for subplots
    num_plots = len(intrusive_thoughts_predicted)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), sharey=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    # Dictionary to store models
    models = {}

    # Loop through each response variable
    for i, intrusive_item in enumerate(intrusive_thoughts_predicted):
        response_variable = data[intrusive_item]  # Extract the response variable data

        # Add a constant for the intercept
        predictors_with_constant = sm.add_constant(mind_wandering_predictors)

        # Fit the regression model
        model = sm.OLS(response_variable, predictors_with_constant).fit()

        # Store the model in the dictionary
        models[intrusive_item] = model

        # Extract p-values
        p_values = model.pvalues.drop('const')  # Exclude the intercept (const)

        # Sort predictors by their numeric prefix (e.g., "(1)", "(2)")
        sorted_indices = sorted(
            range(len(mind_wandering_predictors.columns)),
            key=lambda idx: int(re.search(r'\((\d+)\)', mind_wandering_predictors.columns[idx]).group(1))
        )

        # Apply the sorted order to p-values and predictor names
        sorted_p_values = p_values.iloc[sorted_indices]
        sorted_predictor_names = [mind_wandering_predictors.columns[idx] for idx in sorted_indices]

        # Plot sorted p-values
        axes[i].bar(sorted_predictor_names, sorted_p_values, color='skyblue', edgecolor='black')
        axes[i].axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
        axes[i].set_title(f'P-values for {re.sub(r"^Mini_Item12_", "", intrusive_item)}', fontsize=14)
        axes[i].set_xlabel('Predictors', fontsize=12)
        axes[i].set_ylabel('P-value', fontsize=12 if i == 0 else 0)  # Add ylabel only for the first plot

        # Update X-axis labels with sorted predictor names
        axes[i].set_xticks(range(len(sorted_predictor_names)))
        axes[i].set_xticklabels(sorted_predictor_names, rotation=45, ha='right', fontsize=10)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return models  # Return all models


# Example usage
models = linear_regression_trial(new_df_mixed_genders) 

for state, model_data in models.items():
    print("state", model_data.summary())

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_signi(models, data):
    """
    Plots significant predictors against response variables, including p-values, correlation,
    regression line, and R^2 displayed in the legend.
    """
    for state, model_data in models.items():
        # Extract significant predictors (p-value <= 0.05), excluding the intercept ("const")
        significant_results = [
            predictor for predictor in model_data.pvalues.index
            if predictor != "const" and model_data.pvalues[predictor] <= 0.05
        ]

        # Ensure the state variable exists in the data
        if state not in data.columns:
            print(f"State variable '{state}' not found in data.")
            continue

        # Plot each significant predictor against the state variable
        for predic in significant_results:
            # Ensure predictor exists in the data
            if predic not in data.columns:
                print(f"Predictor '{predic}' not found in data.")
                continue

            # Drop NaN values for correlation and regression calculations
            valid_data = data[[predic, state]].dropna()

            if valid_data.empty:
                print(f"No valid data for predictor '{predic}' and state '{state}'.")
                continue

            # Format p-value to 4 significant figures
            p_value_toprint = f"{model_data.pvalues[predic]:.4g}"

            # Calculate correlation coefficient
            corr_toprint = np.corrcoef(valid_data[predic], valid_data[state])[0, 1]

            # Calculate regression line using linregress
            slope, intercept, r_value, _, _ = linregress(valid_data[predic], valid_data[state])
            x_vals = np.linspace(valid_data[predic].min(), valid_data[predic].max(), 100)
            y_vals = slope * x_vals + intercept

            # Plot scatter and regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(valid_data[predic], valid_data[state], color='blue', alpha=0.6, label='Data')
            plt.plot(
                x_vals, y_vals, color='red',
                label=(
                    f"Regression Line\n"
                    f"$R^2$: {r_value**2:.2f}\n"
                    f"p-value: {p_value_toprint}\n"
                    f"Correlation: {corr_toprint:.2f}"
                )
            )
            
            # Add title and labels
            plt.title(f'Scatter Plot: {predic} vs {state}', fontsize=14)
            plt.xlabel(predic, fontsize=12)
            plt.ylabel(state, fontsize=12)
            plt.grid(True)

            # Add legend with text information
            plt.legend(loc='best', fontsize=10)

            plt.tight_layout()
            plt.show()

plot_signi(models, new_df_mixed_genders)
