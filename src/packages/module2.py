import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_sessions(data):
    """Standardize session-level scores for each participant."""
    session_columns = [col for col in data.columns if '_session' in col or 'Mini_Item' in col]
    for col in session_columns:
        # Ensure numeric data, replace non-numeric values with NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

        # Skip standardization if column is completely empty or NaN
        if data[col].isna().all():
            print(f"Skipping standardization for column: {col} (all values are NaN or non-numeric)")
            continue

        # Perform standardization
        col_mean = data[col].mean()
        col_std = data[col].std()
        data[col + '_standardized'] = (data[col] - col_mean) / col_std
    return data

def meaning_the_sessions(data):
    """Calculate mean for standardized session data."""
    columns = data.columns
    stripped_columns = [re.sub(r'_session\d$', '', col) for col in columns if col != 'sex']
    unique_columns = set(stripped_columns)
    for column_uni in unique_columns:
        matching_columns = [col for col in columns if re.sub(r'_session\d$', '', col) == column_uni and '_standardized' in col]
        numeric_data = data[matching_columns].apply(pd.to_numeric, errors='coerce')
        if not numeric_data.empty:
            row_means = numeric_data.mean(axis=1, skipna=True)
            data[column_uni + '_mean'] = row_means
    return data

def separating_genders(data):
    """Separate data by gender."""
    if 'sex' not in data.columns:
        raise KeyError("'sex' column is missing in the DataFrame.")
    data['sex'] = data['sex'].astype(str).str.strip().str.lower()
    new_df_male = data[data['sex'] == 'm'].copy()
    new_df_female = data[data['sex'] == 'f'].copy()
    return new_df_female, new_df_male

def plot_side_by_side_bars(female_data, male_data, united_data, title):
    """Plot side-by-side comparison bars for gender-separated data."""
    def get_sorted_columns(data):
        mind_wandering_columns = [col for col in data.columns if re.match(r'^\(\d+\)', col) and '_mean' in col]
        mind_wandering_columns.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
        return mind_wandering_columns

    female_columns = get_sorted_columns(female_data)
    male_columns = get_sorted_columns(male_data)
    united_columns = get_sorted_columns(united_data)

    female_means = female_data[female_columns].mean()
    male_means = male_data[male_columns].mean()
    united_means = united_data[united_columns].mean()

    bar_width = 0.25
    x_positions = np.arange(len(female_columns))

    plt.figure(figsize=(14, 7))
    plt.bar(x_positions - bar_width, female_means, width=bar_width, label='Female', alpha=0.7)
    plt.bar(x_positions, male_means, width=bar_width, label='Male', alpha=0.7)
    plt.bar(x_positions + bar_width, united_means, width=bar_width, label='United', alpha=0.7)

    plt.xticks(x_positions, female_columns, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Mind-Wandering Dimensions')
    plt.ylabel('Mean Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_intrusive_thoughts(female_data, male_data, united_data, states_columns, title):
    """Plot intrusive thoughts across states."""
    female_means = female_data[states_columns].mean()
    male_means = male_data[states_columns].mean()
    united_means = united_data[states_columns].mean()

    overall_mean = united_data[states_columns].mean().mean()

    bar_width = 0.25
    x_positions = np.arange(len(states_columns))

    plt.figure(figsize=(14, 7))
    plt.bar(x_positions - bar_width, female_means, width=bar_width, label='Female', alpha=0.7)
    plt.bar(x_positions, male_means, width=bar_width, label='Male', alpha=0.7)
    plt.bar(x_positions + bar_width, united_means, width=bar_width, label='United', alpha=0.7)

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
    """Compare grouped session data."""
    def custom_sort_key(column_name):
        match = re.match(r'\((\d+)\)', column_name)
        return int(match.group(1)) if match else float('inf')

    session_columns = sorted([col for col in data.columns if col.endswith('_standardized')], key=custom_sort_key)
    overall_means = []

    for col in session_columns:
        session_data = data[col].apply(pd.to_numeric, errors='coerce')
        overall_means.append(session_data.mean())

    plt.figure(figsize=(16, 8))
    plt.plot(overall_means, color='red', marker='o', linewidth=2, label='Overall Mean')
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

df = pd.read_csv('/Users/mayacohen/Desktop/project_gadol/data/participants.the.one.that.works.csv')
columns = [
    '(1)Discontinuity of Mind_session1', '(2)Theory of Mind_session1', '(3)Self_session1', '(4)Planning_session1',
    '(5)Sleepiness_session1', '(6)Comfort_session1', '(7)Somatic Awareness_session1', '(8)Health Concern_session1',
    '(9)Visual Thought_session1', '(10)Verbal Thought_session1', '(1)Discontinuity of Mind_session2',
    '(2)Theory of Mind_session2', '(3)Self_session2', '(4)Planning_session2', '(5)Sleepiness_session2',
    '(6)Comfort_session2', '(7)Somatic Awareness_session2', '(8)Health Concern_session2', '(9)Visual Thought_session2',
    '(10)Verbal Thought_session2', '(1)Discontinuity of Mind_session3', '(2)Theory of Mind_session3',
    '(3)Self_session3', '(4)Planning_session3', '(5)Sleepiness_session3', '(6)Comfort_session3',
    '(7)Somatic Awareness_session3', '(8)Health Concern_session3', '(9)Visual Thought_session3',
    '(10)Verbal Thought_session3', "Mini_Item12_EO1", "Mini_Item12_EC1", "Mini_Item12_Music1", "Mini_Item12_Memory1",
    "Mini_Item12_Subtraction1", "Mini_Item12_EO2", "Mini_Item12_EC2", "Mini_Item12_Music2", "Mini_Item12_Memory2",
    "Mini_Item12_Subtraction2", "Mini_Item12_EO3", "Mini_Item12_EC3", "Mini_Item12_Music3", "Mini_Item12_Memory3",
    "Mini_Item12_Subtraction3", 'sex'
]

new_df_mixed_genders = df[columns].copy()
new_df_mixed_genders = standardize_sessions(new_df_mixed_genders)
new_df_mixed_genders = meaning_the_sessions(new_df_mixed_genders)
new_df_female, new_df_male = separating_genders(new_df_mixed_genders)

plot_side_by_side_bars(new_df_female, new_df_male, new_df_mixed_genders, "Side-by-Side Comparison")
plot_intrusive_thoughts(
    new_df_female,
    new_df_male,
    new_df_mixed_genders,
    ["Mini_Item12_EO1_standardized", "Mini_Item12_EC1_standardized"],
    "Intrusive Thoughts Across States"
)
compare_sessions_grouped(new_df_mixed_genders, "Comparison of Sessions")
