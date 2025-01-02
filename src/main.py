import pandas as pd 
import re 
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.packages.new_data_frames import meaning_the_sessions as mts
from src.packages.new_data_frames import separating_genders 

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
mts(new_df_mixed_genders)

# Call the function
new_df_female, new_df_male = separating_genders(new_df_mixed_genders)

# Print results
print("Female DataFrame:")
print(new_df_female)

print("\nMale DataFrame:")
print(new_df_male)
 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming `data` is your dataframe and contains questionnaire columns
columns_of_interest = [
    "(1)Discontinuity of Mind_session1",
    "(2)Theory of Mind_session1",
    "(3)Self_session1",
    "(4)Planning_session1",
    "(5)Sleepiness_session1",
    "(6)Comfort_session1",
    "(7)Somatic Awareness_session1",
    "(8)Health Concern_session1",
    "(9)Visual Thought_session1",
    "(10)Verbal Thought_session1"
]

