import pandas as pd 
from src.packages.tools.new_data_frames import meaning_the_sessions as mts
from src.packages.tools.new_data_frames import remove_outliers as remove_outliers
from src.packages.tools.explorations import plot_side_by_side_bars as plot_side_by_side_bars
from src.packages.tools.explorations import plot_intrusive_thoughts as plot_intrusive_thoughts
from src.packages.tools.explorations import compare_sessions_grouped as compare_sessions_grouped
from src.packages.tools.explorations import plot_correlation_matrix as plot_correlation_matrix
from src.packages.tools.anlysis import linear_regression_trial as linear_regression_trial
from src.packages.tools.anlysis import plot_signi as plot_signi
from src.packages.tools.anlysis import linear_regression_with_sex_interactions as linear_regression_with_sex_interactions
from src.packages.tools.anlysis import plot_signi_bysex as plot_signi_bysex
from src.packages.tools.new_data_frames import separating_genders 
path='/Users/mayacohen/Desktop/project_gadol/data/participants.the.one.that.works.csv'

df = pd.read_csv(path)



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
new_df_mixed_genders = remove_outliers(new_df_mixed_genders)
mts(new_df_mixed_genders)

# Call the function
new_df_female, new_df_male = separating_genders(new_df_mixed_genders)

# Print results
print("Female DataFrame:")
print(new_df_female)

print("\nMale DataFrame:")
print(new_df_male)
 
plot_side_by_side_bars(new_df_female, new_df_male, new_df_mixed_genders, "Side-by-Side Comparison of Mind-Wandering Dimensions")

plot_intrusive_thoughts(
    new_df_female,
    new_df_male,
    new_df_mixed_genders,
    )
 
compare_sessions_grouped(
  new_df_mixed_genders, 
 "Comparison of Sessions with Max-Min Differences"
)

plot_correlation_matrix(new_df_mixed_genders, title="Custom Ordered Correlation Heatmap")

models = linear_regression_trial(new_df_mixed_genders) 

plot_signi(models, new_df_mixed_genders)

models_sex, new_df_mixed_genders_proccessed= linear_regression_with_sex_interactions(new_df_mixed_genders)

plot_signi_bysex(models_sex, new_df_mixed_genders, new_df_male, new_df_female)
