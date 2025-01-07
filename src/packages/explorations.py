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