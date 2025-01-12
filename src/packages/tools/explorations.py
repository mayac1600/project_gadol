import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plot_side_by_side_bars(female_data, male_data, united_data, title="Side-by-Side Comparison of Mind-Wandering Dimensions"):
    """
    Create a side-by-side bar chart comparing the mean scores of mind-wandering dimensions
    across female, male, and united datasets.
    """
    # Extract and sort columns for mind-wandering dimensions in each dataset
    datasets = [female_data, male_data, united_data]
    sorted_columns = []

    for data in datasets:
        # Identify columns that represent mind-wandering dimensions
        mind_wandering_columns = [
            col for col in data.columns
            if re.match(r'^\(\d+\)', col) and '_mean' in col
        ]
        # Sort columns numerically based on the dimension number
        mind_wandering_columns.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
        sorted_columns.append(mind_wandering_columns)

    #  Unpack sorted columns and ensure consistency across datasets
    female_columns, male_columns, united_columns = sorted_columns
    assert female_columns == male_columns == united_columns, "Column orders do not match!"

    # Convert columns to numeric for all datasets
    female_data = female_data[female_columns].apply(pd.to_numeric, errors='coerce')
    male_data = male_data[male_columns].apply(pd.to_numeric, errors='coerce')
    united_data = united_data[united_columns].apply(pd.to_numeric, errors='coerce')

    #  Calculate means and standard deviations for each group
    female_means = female_data.mean()
    female_stds = female_data.std()

    male_means = male_data.mean()
    male_stds = male_data.std()

    united_means = united_data.mean()
    united_stds = united_data.std()

    # : Set up x-axis positions for the bars
    n_dimensions = len(female_columns)
    bar_width = 0.25
    x_positions = np.arange(n_dimensions)

    female_positions = x_positions - bar_width
    male_positions = x_positions
    united_positions = x_positions + bar_width

    #  Create the bar chart
    plt.figure(figsize=(14, 7))
    plt.bar(female_positions, female_means, yerr=female_stds, width=bar_width, label='Female', color='midnightblue', alpha=0.7)
    plt.bar(male_positions, male_means, yerr=male_stds, width=bar_width, label='Male', color='moccasin', alpha=0.7)
    plt.bar(united_positions, united_means, yerr=united_stds, width=bar_width, label='United', color='teal', alpha=0.7)

    # Step 7: Style the chart
    plt.xticks(x_positions, female_columns, rotation=45, ha='right')  # Label bars with column names
    plt.title(title)
    plt.xlabel('Mind-Wandering Dimensions')
    plt.ylabel('Mean Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()

def compare_sessions_grouped(data, title):
    """
    This function analyzes and visualizes session-based data by identifying metrics with a common base name across session-specific columns (e.g., metric1, metric2), computing session means, max-min differences, and overall means. It generates a grouped bar chart to compare session means, a line plot for max-min differences, and overlays dashed lines for overall means with annotations. The function provides a dictionary summarizing these statistics for further analysis, offering insights into trends and variability across sessions.
    """

    # Extract unique numeric suffixes representing session numbers
    try:
        session_numbers = sorted(set(
            int(re.search(r'(\d+)$', col).group(1))
            for col in data.columns if re.search(r'(\d+)$', col)
        ))
    except AttributeError:
        print("Error: Column names must end with numeric session identifiers.")
        return

    # Identify common base names across all sessions
    base_names = set(
        re.sub(r'\d+$', '', col) for col in data.columns
    )
    common_bases = [
        base for base in base_names
        if all(f"{base}{num}" in data.columns for num in session_numbers)
    ]

    if not common_bases:
        print("No common base metrics found across all sessions.")
        return

    # Initialize lists for plotting
    session_means = {num: [] for num in session_numbers}
    max_min_differences = []
    overall_means = []
    group_labels = []

    for base in common_bases:
        # Collect data for all sessions of the current base
        session_data = pd.DataFrame({
            num: pd.to_numeric(data[f"{base}{num}"], errors='coerce')
            for num in session_numbers
        })

        # Skip if all data is NaN
        if session_data.isnull().all().all():
            continue

        # Calculate means for each session
        for num in session_numbers:
            session_means[num].append(session_data[num].mean(skipna=True))

        # Calculate overall mean and max-min difference
        overall_mean = session_data.mean(axis=1, skipna=True).mean()
        max_min_diff = (session_data.max(axis=1) - session_data.min(axis=1)).mean()

        overall_means.append(overall_mean)
        max_min_differences.append(max_min_diff)
        group_labels.append(base)

    # If no valid data is found, exit the function
    if not group_labels:
        print("No valid data found for plotting.")
        return

    # Plotting
    x_positions = np.arange(len(group_labels))
    bar_width = 0.2

    plt.figure(figsize=(16, 8))

    # Plot bars for each session
    for i, num in enumerate(session_numbers):
        plt.bar(
            x_positions + i * bar_width,
            session_means[num],
            width=bar_width,
            label=f'Session {num}'
        )

    # Plot line for max-min differences
    plt.plot(
        x_positions + bar_width * (len(session_numbers) - 1) / 2,
        max_min_differences,
        color='red',
        marker='o',
        linewidth=2,
        label='Max-Min Difference'
    )

    # Add overall mean lines and annotations
    for i, mean in enumerate(overall_means):
        plt.plot(
            [x_positions[i] - bar_width, x_positions[i] + bar_width * len(session_numbers)],
            [mean, mean],
            color='black',
            linestyle='--',
            linewidth=1.5
        )
        plt.text(
            x_positions[i] + bar_width * (len(session_numbers) - 1) / 2,
            mean + 0.05,
            f"{mean:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            fontweight='bold'
        )

    # Formatting
    plt.xticks(
        x_positions + bar_width * (len(session_numbers) - 1) / 2,
        group_labels,
        rotation=45,
        ha='right'
    )
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.axhline(
        np.mean(max_min_differences),
        color='purple',
        linestyle='--',
        label=f'Avg Max-Min = {np.mean(max_min_differences):.2f}'
    )
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_intrusive_thoughts(female_data, male_data, united_data, title="Intrusive Thoughts Across States (Men vs Women vs All)"):
    ''''This function visualizes the comparison of mean intrusive thought scores (and their standard deviations) across female, male, and united groups for shared states (columns matching "Mini_Item12" with "mean"). It calculates group-wise means, standard deviations, and an overall mean, plotting grouped bar charts for each category with error bars and a dashed horizontal line for the overall mean. The x-axis represents the shared states, and the y-axis shows mean scores, with formatting for clarity and readability.'''
    # Identify common columns across all groups
    states_columns = [col for col in united_data.columns if "Mini_Item12" in col and "mean" in col]
    states_columns = [
        col for col in states_columns
        if col in female_data.columns and col in male_data.columns
    ]

    # Check if there are no common columns
    if not states_columns:
        print("No common columns found across groups. Skipping plot.")
        return  # Exit the function gracefully

    # Ensure numeric data only
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

def plot_correlation_matrix(data, variables=None, title="Correlation Heatmap"):
    """
    Plots a correlation matrix heatmap for the specified variables in a custom order.

    Parameters:
    - data (DataFrame): The dataset containing the variables.
    - variables (list or None): A list of variable names to include in the correlation matrix.
      If None, defaults to columns containing 'mean' in their names.
    - title (str): Title of the heatmap.
    """
    # Ensure `data` is a valid DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The `data` parameter must be a pandas DataFrame.")

    # Determine variables if not provided
    if variables is None:
        variables = [col for col in data.columns if "mean" in col]

    # Handle case where no valid variables are found
    if not variables:
        raise ValueError("No valid variables found to include in the correlation matrix.")

    # Custom sorting logic inline
    variables = sorted(
        variables, 
        key=lambda column_name: int(re.match(r'\((\d+)\)', column_name).group(1)) 
        if re.match(r'\((\d+)\)', column_name) else float('inf')
    )

    # Ensure variables exist in the DataFrame
    missing_vars = [var for var in variables if var not in data.columns]
    if missing_vars:
        raise KeyError(f"The following variables are missing from the DataFrame: {missing_vars}")

    # Ensure variables have numeric data
    non_numeric_vars = [var for var in variables if not pd.api.types.is_numeric_dtype(data[var])]
    if non_numeric_vars:
        raise ValueError(f"The following variables are not numeric and cannot be used: {non_numeric_vars}")

    # Compute the correlation matrix
    correlation_matrix = data[variables].corr()

    # Handle edge case where correlation matrix is empty
    if correlation_matrix.empty:
        raise ValueError("The correlation matrix is empty. Ensure variables contain valid numeric data.")

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xticks(ticks=np.arange(len(variables)), labels=variables, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(variables)), labels=variables, rotation=0)
    plt.tight_layout()
    plt.show()