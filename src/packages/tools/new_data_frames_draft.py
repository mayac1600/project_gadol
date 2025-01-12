import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress

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

new_df_mixed_genders= df[columns].copy()
# Select the columns
def remove_outliers(data):
    """
    Replace outliers (values outside 1.5 times the IQR) with NaN.
    """
    for column in data.select_dtypes(include=[np.number]):  # Only process numeric columns
        Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
        Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range

        # Define lower and upper bounds for outliers
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        
        # Replace outliers with NaN
        data[column] = data[column].apply(lambda x: np.nan if x < lower_limit or x > upper_limit else x)

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


new_df_mixed_genders = meaning_the_sessions(new_df_mixed_genders)
print (new_df_mixed_genders.head())



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
new_df_female, new_df_male = separating_genders(new_df_mixed_genders)


def plot_side_by_side_bars(female_data, male_data, united_data, title="Side-by-Side Comparison of Mind-Wandering Dimensions"):
    def get_sorted_columns(data):
        mind_wandering_columns = [
            col for col in data.columns
            if re.match(r'^\(\d+\)', col) and '_mean' in col
        ]
        mind_wandering_columns.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
        return mind_wandering_columns

    female_columns = get_sorted_columns(female_data)
    male_columns = get_sorted_columns(male_data)
    united_columns = get_sorted_columns(united_data)

    assert female_columns == male_columns == united_columns, "Column orders do not match!"

    # Calculate statistics for each group (NaNs are automatically skipped)
    female_means = female_data[female_columns].mean()
    female_stds = female_data[female_columns].std()

    male_means = male_data[male_columns].mean()
    male_stds = male_data[male_columns].std()

    united_means = united_data[united_columns].mean()
    united_stds = united_data[united_columns].std()

    # X positions for the bars
    n_dimensions = len(female_columns)
    bar_width = 0.25
    x_positions = np.arange(n_dimensions)

    female_positions = x_positions - bar_width
    male_positions = x_positions
    united_positions = x_positions + bar_width

    # Plot
    plt.figure(figsize=(14, 7))
    plt.bar(female_positions, female_means, yerr=female_stds, width=bar_width, label='Female', color='midnightblue', alpha=0.7)
    plt.bar(male_positions, male_means, yerr=male_stds, width=bar_width, label='Male', color='moccasin', alpha=0.7)
    plt.bar(united_positions, united_means, yerr=united_stds, width=bar_width, label='United', color='teal', alpha=0.7)

    plt.xticks(x_positions, female_columns, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Mind-Wandering Dimensions')
    plt.ylabel('Mean Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()

plot_side_by_side_bars(new_df_female, new_df_male, new_df_mixed_genders)

def plot_intrusive_thoughts(female_data, male_data, united_data, title= "Intrusive Thoughts Across States (Men vs Women vs All)"):
    # Ensure numeric data only
    states_columns=[col for col in united_data.columns if "Mini_Item12" in col and "mean" in col]
    female_data = female_data[states_columns].apply(pd.to_numeric, errors='coerce')
    male_data = male_data[states_columns].apply(pd.to_numeric, errors='coerce')
    united_data = united_data[states_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate means and standard deviations for each state (NaNs are skipped)
    female_means = female_data.mean()
    female_stds = female_data.std()

    male_means = male_data.mean()
    male_stds = male_data.std()

    united_means = united_data.mean()
    united_stds = united_data.std()

    # Calculate the overall mean across all states for the horizontal line
    overall_mean = united_data.mean().mean()

    # X positions for the bars
    n_states = len(states_columns)
    bar_width = 0.25
    x_positions = np.arange(n_states)

    female_positions = x_positions - bar_width
    male_positions = x_positions
    united_positions = x_positions + bar_width

    # Plot
    plt.figure(figsize=(14, 7))

    # Bar plots for each group
    plt.bar(female_positions, female_means, yerr=female_stds, width=bar_width, label='Female', color='mediumpurple', alpha=0.7)
    plt.bar(male_positions, male_means, yerr=male_stds, width=bar_width, label='Male', color='cadetblue', alpha=0.7)
    plt.bar(united_positions, united_means, yerr=united_stds, width=bar_width, label='United', color='slategray', alpha=0.7)

    # Formatting the plot
    plt.axhline(overall_mean, color='midnightblue', linestyle='--', label='Overall Mean')
    plt.xticks(x_positions, states_columns, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('States')
    plt.ylabel('Mean Intrusive Thoughts')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()


plot_intrusive_thoughts(
    new_df_female,
    new_df_male,
    new_df_mixed_genders,
    )
def compare_sessions_grouped(data, title):
    # Custom sorting logic for ordering columns
    def custom_sort_key(column_name):
        import re
        match = re.match(r'\((\d+)\)', column_name)  # Look for numbers in parentheses at the start
        if match:
            return int(match.group(1))  # Sort numerically by the number
        return float('inf')  # Non-matching columns are placed at the end

    # Group columns based on the last digit (1, 2, 3)
    session1_columns = sorted([col for col in data.columns if col.endswith('1')], key=custom_sort_key)
    session2_columns = sorted([col for col in data.columns if col.endswith('2')], key=custom_sort_key)
    session3_columns = sorted([col for col in data.columns if col.endswith('3')], key=custom_sort_key)

    # Ensure common bases exist across all sessions
    common_bases = sorted(set(
        col[:-1] for col in session1_columns
        if col[:-1] in [c[:-1] for c in session2_columns] and col[:-1] in [c[:-1] for c in session3_columns]
    ), key=custom_sort_key)

    max_min_differences = []
    group_labels = []

    session1_means, session2_means, session3_means = [], [], []
    overall_means = []

    for base in common_bases:
        # Collect columns for this group
        session_columns = [f"{base}{i}" for i in range(1, 4)]
        session_data = data[session_columns].apply(pd.to_numeric, errors='coerce')

        # Calculate means for each session
        session1_means.append(session_data.iloc[:, 0].mean())
        session2_means.append(session_data.iloc[:, 1].mean())
        session3_means.append(session_data.iloc[:, 2].mean())

        # Calculate overall mean for the item
        overall_means.append(session_data.mean(axis=1).mean())

        # Calculate max-min difference
        max_values = session_data.max(axis=1)
        min_values = session_data.min(axis=1)
        max_min_differences.append((max_values - min_values).mean())
        group_labels.append(base)

    # Plotting
    x_positions = np.arange(len(common_bases))
    bar_width = 0.25

    plt.figure(figsize=(16, 8))

    # Bar plots for session means
    plt.bar(x_positions - bar_width, session1_means, width=bar_width, label='Session 1', color='skyblue', alpha=0.7)
    plt.bar(x_positions, session2_means, width=bar_width, label='Session 2', color='orange', alpha=0.7)
    plt.bar(x_positions + bar_width, session3_means, width=bar_width, label='Session 3', color='green', alpha=0.7)

    # Line plot for max-min differences
    plt.plot(x_positions, max_min_differences, color='red', marker='o', linewidth=2, label='Max-Min Difference')

    # Add overall mean horizontal lines and text
    for i, mean in enumerate(overall_means):
        # Horizontal line connecting all bars for the same group
        plt.plot([x_positions[i] - bar_width, x_positions[i] + bar_width], [mean, mean], color='black', linestyle='-', linewidth=1.5)

        # Add mean value text
        plt.text(x_positions[i], mean + 0.05, f"{mean:.2f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    # Formatting
    plt.xticks(x_positions, group_labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.axhline(np.mean(max_min_differences), color='purple', linestyle='--', label=f'Avg Max-Min = {np.mean(max_min_differences):.2f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()

compare_sessions_grouped(
  new_df_mixed_genders, 
 "Comparison of Sessions with Max-Min Differences"
)

import re

def plot_correlation_matrix(data, variables=None, title="Correlation Heatmap"):
    """
    Plots a correlation matrix heatmap for the specified variables in a custom order.

    Parameters:
    - data (DataFrame): The dataset containing the variables.
    - variables (list or None): A list of variable names to include in the correlation matrix.
      If None, defaults to columns containing 'mean' in their names.
    - title (str): Title of the heatmap.
    """
    # Determine variables if not provided
    if variables is None:
        variables = [col for col in data.columns if "mean" in col]

    # Custom sorting logic
    def custom_sort_key(column_name):
        match = re.match(r'\((\d+)\)', column_name)  # Look for numbers in parentheses at the start
        if match:
            return int(match.group(1))  # Sort numerically by the number
        return float('inf')  # Non-matching columns are placed at the end

    variables = sorted(variables, key=custom_sort_key)

    # Compute the correlation matrix
    correlation_matrix = data[variables].corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xticks(ticks=np.arange(len(variables)), labels=variables, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(variables)), labels=variables, rotation=0)
    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_matrix(new_df_mixed_genders, title="Custom Ordered Correlation Heatmap")




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
 
def linear_regression_with_sex_interactions(data):
    """
    Processes data, creates interaction terms for sex, fits regression models, and visualizes results.
    
    Args:
        data (DataFrame): Input dataset containing predictors, targets, and a binary 'sex' column.
    
    Returns:
        models (dict): A dictionary of fitted regression models for each target variable.
        data (DataFrame): The processed DataFrame with interaction terms included.
    """
    # Make a copy of the data to avoid overwriting
    data = data.copy()

    # Identify all relevant columns based on '_mean'
    all_relevant_cols = [col for col in data.columns if "_mean" in col]

    # Replace NaNs and infinities in predictors
    for col in all_relevant_cols:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        data[col] = data[col].fillna(data[col].mean())

    # Encode 'sex' as 0 (male) and 1 (female)
    data['sex'] = data['sex'].map({'m': 0, 'f': 1})

    # Create interaction terms for each predictor
    for predictor in all_relevant_cols:
        interaction_term = f"{predictor}_bysex"
        data[interaction_term] = data[predictor] * data['sex']

    # Separate predictors and target variables
    mind_wandering_predictors = [col for col in all_relevant_cols if "Mini_Item" not in col]
    intrusive_thoughts_predicted = [col for col in all_relevant_cols if "Mini_Item" in col]

    # Initialize dictionary to store models
    models = {}

    # Set up a grid for subplots
    num_plots = len(intrusive_thoughts_predicted)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), sharey=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    # Loop through each target variable
    for i, target in enumerate(intrusive_thoughts_predicted):
        # Combine predictors and their interaction terms
        predictors_with_interactions = mind_wandering_predictors + [f"{pred}_bysex" for pred in mind_wandering_predictors]
        X = data[predictors_with_interactions]
        X = sm.add_constant(X)  # Add constant for the intercept
        y = data[target]

        # Filter rows with valid data
        valid_indices = X.notnull().all(axis=1) & y.notnull()
        X = X[valid_indices]
        y = y[valid_indices]

        # Fit regression model
        try:
            model = sm.OLS(y, X).fit()
            models[target] = model

            # Extract p-values for interaction terms
            interaction_p_values = model.pvalues[[col for col in X.columns if "_bysex" in col]]

            # Sort interaction terms by their numeric prefix (if present)
            sorted_indices = sorted(
                range(len(interaction_p_values)),
                key=lambda idx: int(re.search(r'\((\d+)\)', interaction_p_values.index[idx]).group(1))
            )
            
            # Sort interaction p-values and names
            sorted_p_values = interaction_p_values.iloc[sorted_indices]
            sorted_interaction_names = [interaction_p_values.index[idx] for idx in sorted_indices]

            # Plot sorted interaction p-values
            axes[i].bar(sorted_interaction_names, sorted_p_values, color='skyblue', edgecolor='black')
            axes[i].axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
            axes[i].set_title(f'P-values for {re.sub(r"^Mini_Item12_", "", target)}', fontsize=14)
            axes[i].set_xlabel('Interaction Predictors (_bysex)', fontsize=12)
            axes[i].set_ylabel('P-value', fontsize=12 if i == 0 else 0)  # Add ylabel only for the first plot
            axes[i].set_xticks(range(len(sorted_interaction_names)))
            axes[i].set_xticklabels(sorted_interaction_names, rotation=45, ha='right', fontsize=10)
            axes[i].legend()

        except Exception as e:
            print(f"Error fitting model for '{target}': {e}")

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return models, data

models_sex, new_df_mixed_genders_proccessed= linear_regression_with_sex_interactions(new_df_mixed_genders)

from scipy.stats import linregress


def plot_signi_bysex(models, data, maledata, femaledata):
    """
    Plots significant predictors (_bysex) against state variables for male and female datasets together.
    Shows the p-value for the interaction term and the correlation coefficient for each gender.

    Args:
        models (dict): Dictionary of fitted regression models.
        data (DataFrame): Full dataset.
        maledata (DataFrame): Dataset filtered for males.
        femaledata (DataFrame): Dataset filtered for females.
    """
    all_significant_results = {}

    for state, model_data in models.items():
        # Extract significant interaction terms (_bysex)
        significant_results_columns = [
            predictor for predictor in model_data.pvalues.index
            if predictor != "const" and model_data.pvalues[predictor] <= 0.05 and "_bysex" in predictor
        ]
        all_significant_results[state] = significant_results_columns

        for predictor in significant_results_columns:
            # Strip "_bysex" to get the base predictor
            base_predictor = predictor.replace("_bysex", "")
            p_value_for_interaction = model_data.pvalues[predictor]

            # Ensure base predictor and state columns exist in both male and female data
            if base_predictor not in maledata.columns or state not in maledata.columns:
                print(f"Skipping {base_predictor} or {state} for males - columns not found.")
                continue
            if base_predictor not in femaledata.columns or state not in femaledata.columns:
                print(f"Skipping {base_predictor} or {state} for females - columns not found.")
                continue

            # Combine male and female data for the plot
            male_data = maledata[[base_predictor, state]].dropna()
            female_data = femaledata[[base_predictor, state]].dropna()

            if male_data.empty or female_data.empty:
                print(f"No valid data for {base_predictor} and {state}.")
                continue

            # Calculate correlation and regression line for males
            corr_male = np.corrcoef(male_data[base_predictor], male_data[state])[0, 1]
            slope_male, intercept_male, r_value_male, _, _ = linregress(male_data[base_predictor], male_data[state])
            x_vals_male = np.linspace(male_data[base_predictor].min(), male_data[base_predictor].max(), 100)
            y_vals_male = slope_male * x_vals_male + intercept_male

            # Calculate correlation and regression line for females
            corr_female = np.corrcoef(female_data[base_predictor], female_data[state])[0, 1]
            slope_female, intercept_female, r_value_female, _, _ = linregress(female_data[base_predictor], female_data[state])
            x_vals_female = np.linspace(female_data[base_predictor].min(), female_data[base_predictor].max(), 100)
            y_vals_female = slope_female * x_vals_female + intercept_female

            # Plot male and female data together
            plt.figure(figsize=(10, 6))
            plt.scatter(
                male_data[base_predictor], male_data[state], alpha=0.6, label=f'Male Data (corr: {corr_male:.2f})',
                marker='o', color='blue'
            )
            plt.plot(x_vals_male, y_vals_male, color='blue', linestyle='--', label='Male Regression Line')

            plt.scatter(
                female_data[base_predictor], female_data[state], alpha=0.6, label=f'Female Data (corr: {corr_female:.2f})',
                marker='s', color='orange'
            )
            plt.plot(x_vals_female, y_vals_female, color='orange', linestyle='--', label='Female Regression Line')

            # Add title, labels, legend, and p-value for interaction
            plt.title(f'{base_predictor} vs {state} (Interaction p-value: {p_value_for_interaction:.4g})', fontsize=14)
            plt.xlabel(base_predictor, fontsize=12)
            plt.ylabel(state, fontsize=12)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()

    print("All significant results by state:", all_significant_results)


# Call the function with your models and dataset
plot_signi_bysex(models_sex, new_df_mixed_genders, new_df_male, new_df_female)

