import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load the dataset
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
    "Mini_Item12_EO3", "Mini_Item12_EC3", "Mini_Item12_Music3", "Mini_Item12_Memory3", "Mini_Item12_Subtraction3", 'sex'
]

# Select the columns
new_df = df[columns].copy()

# Define the target states and predictors
target_states = [
    "Mini_Item12_EO1", "Mini_Item12_EC1", "Mini_Item12_Music1", "Mini_Item12_Memory1", "Mini_Item12_Subtraction1"
]
predictors = [
    '(1)Discontinuity of Mind_session1', '(2)Theory of Mind_session1', '(3)Self_session1', '(4)Planning_session1',
    '(5)Sleepiness_session1', '(6)Comfort_session1', '(7)Somatic Awareness_session1', '(8)Health Concern_session1',
    '(9)Visual Thought_session1', '(10)Verbal Thought_session1'
]

# Clean and preprocess the data
new_df[target_states + predictors] = new_df[target_states + predictors].apply(pd.to_numeric, errors='coerce')
new_df = new_df.dropna()

# Separate data by gender
male_data = new_df[new_df['sex'].str.lower() == 'm']
female_data = new_df[new_df['sex'].str.lower() == 'f']

# Function to explore data
def explore_data(df, target_states, predictors):
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[predictors + target_states].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Pair plots
    for target in target_states:
        sns.pairplot(df[predictors + [target]], diag_kind='kde')
        plt.suptitle(f'Pairwise Relationships with {target}', y=1.02)
        plt.show()

    # Distribution of intrusive thoughts
    for target in target_states:
        sns.histplot(df[target], kde=True)
        plt.title(f'Distribution of {target}')
        plt.show()

# Explore data for both genders
print("Exploring male data...")
explore_data(male_data, target_states, predictors)
print("Exploring female data...")
explore_data(female_data, target_states, predictors)

# Function to train predictive models
def train_predictive_models(df, target_states, predictors):
    results = {}
    scaler = StandardScaler()

    for target in target_states:
        print(f"Training model for {target}...")
        X = scaler.fit_transform(df[predictors])
        y = (df[target] > df[target].median()).astype(int)  # Binary classification

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC for {target}: {auc:.2f}")

        # Feature importance
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_indices], align="center")
        plt.xticks(range(len(importances)), np.array(predictors)[sorted_indices], rotation=45, ha='right')
        plt.title(f"Feature Importance for {target}")
        plt.show()

        results[target] = auc

    return results

# Train models for male and female data
print("Training models for male data...")
male_results = train_predictive_models(male_data, target_states, predictors)

print("Training models for female data...")
female_results = train_predictive_models(female_data, target_states, predictors)

# Compare results
print("Comparing male and female results:")
comparison = pd.DataFrame({"Male AUC": male_results, "Female AUC": female_results})
print(comparison)
comparison.plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Predictive Performance by Gender")
plt.ylabel("AUC-ROC")
plt.xlabel("Target States")
plt.legend()
plt.tight_layout()
plt.show()
