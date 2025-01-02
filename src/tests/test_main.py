import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_side_by_side_bars(female_data, male_data, united_data, title):
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

plot_side_by_side_bars(new_df_female, new_df_male, new_df_mixed_genders, "Side-by-Side Comparison of Mind-Wandering Dimensions")

def plot_intrusive_thoughts(female_data, male_data, united_data, states_columns, title):
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


plot_intrusive_thoughts(
    new_df_female,
    new_df_male,
    new_df_mixed_genders,
    ["Mini_Item12_EO1", "Mini_Item12_EC1", "Mini_Item12_Music1", "Mini_Item12_Memory1", "Mini_Item12_Subtraction1"],
    "Intrusive Thoughts Across States (Men vs Women vs All)"
    )
def compare_sessions_grouped(data, title):
    # Group columns based on the last digit (1, 2, 3)
    session1_columns = [col for col in data.columns if col.endswith('1')]
    session2_columns = [col for col in data.columns if col.endswith('2')]
    session3_columns = [col for col in data.columns if col.endswith('3')]

    # Ensure common bases exist across all sessions
    common_bases = sorted(set(
        col[:-1] for col in session1_columns
        if col[:-1] in [c[:-1] for c in session2_columns] and col[:-1] in [c[:-1] for c in session3_columns]
    ))

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
# Correlation Analysis
plt.figure(figsize=(10, 8))
correlation_matrix = new_df_mixed_genders.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
       

