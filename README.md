Welcome our the project !!!
## BACKGROUND TO THIS PROJECT
   This project investigates how different aspects of mind-wandering predict intrusive thoughts across various cognitive tasks measured by an after imagining questionnaire and explores whether these predictions differ between men and women. The cognitive tasks took place under EEG measurements.
   The dataset includes 10 mind-wandering dimensions collected in each of three sessions.
   After every of the five cognitive tasks (Eyes-Open, Eyes-Closed, Music, Memory, and Subtraction), participants answered the Mini NYC-Q questionnaire, measuring various mental processes, we will focus on (item 12) intrusive thoughts.
   By aggregating the scores from each of the three sessions into mean value, we will analyze which dimensions are the strongest predictors of intrusive thoughts and assess gender-specific patterns. If direct predictions are unclear, we will explore indirect impacts on mood or other variables, contributing to the refinement of experimental designs and understanding of cognitive processes.




    OUR RESEARCH QUESTION:
        How different dimensions of mind-wandering predict intrusive thoughts across various cognitive tasks. Furthermore, it investigates whether these relationships vary between male and female participants.




##### HOW TO USE THIS PROJECT'S SCIRIPTS #####
## PROJECT SOURCES
    The article link:
        https://www.nature.com/articles/s41597-022-01607-9


    link to Project's raw data:
        https://openneuro.org/datasets/ds004148/versions/1.0.1/file-display/participants.tsv


    Project Workflow and Documentation:
        https://docs.google.com/document/d/1UivLB8QXj9JdQKjBjBoU3abGYvDbTCck/edit


    Project Report:
        https://docs.google.com/document/d/1jbuRMIA-MGc_SwT2X8NXAzK3iP2b9Vxe/edit
## 1. GENERAL AND PREWORKING SETS
   make sure you have the correct python version 3.13.0 by writing in the terminal python --version
## 2. VIRTUAL ENVIROMENT ACTIVATION
First when working on this project install and activate a virtual enviroment to make sure there is a consistency, and avoid conflicts with system packages outside of this project.
1) pip install virtualenv
2) python -m venv venv
3) venv\Scripts\activate
## 3. PIP INSTALL RELEVANT PACKAGES
   To use the project properly with all its libraries,
   run in the terminal (in the venv):
   1) python.exe -m pip install --upgrade pip
   2) pip install -e .
   3) pip install -e .[dev]

   pip list,
    then the output in the terminal should be:
    Package         Version
--------------- -----------
contourpy                         1.3.1
cycler                            0.12.1
fonttools                         4.55.3
kiwisolver                        1.4.8
matplotlib                        3.10.0
mind-wandering-intrusive-thoughts 1.0.0
mypy                              1.14.1
mypy-extensions                   1.0.0
numpy                             2.2.1
packaging                         24.2
pandas                            2.2.3
pandas-stubs                      2.2.3.241126
patsy                             1.0.1
pillow                            11.1.0
pip                               24.2
pyparsing                         3.2.1
python-dateutil                   2.9.0.post0
pytz                              2024.2
scipy                             1.15.1
seaborn                           0.13.2
six                               1.17.0
statsmodels                       0.14.4
types-pytz                        2024.2.0.20241221
types-seaborn                     0.13.2.20250111
typing_extensions                 4.12.2
tzdata    
## 4. DEFINE CORRECT PATH
   1. To use this project you will first need to open the Data Set this project working on,
   in order to do this you need to copy the csv file path from the data folder by pressing it with the right click and choose "Copy Path".
   2. After copy path the you need to paste it in the main.py file,
       it will look like this, in the 23 line of th code:
            path = "PASTE\YOUR\PATH\HERE\participants.the.one.that.works.csv"
            df = pd.read_csv( path)
       dont forget to put "" before and after
       if it cant read the file add r"" too look like this: r"YOUR\PATH"
## 5. UNDERSTANDING PROJECT STRUCTURE
project_gadol
   |
   |-----data
   |       |
   |       |------THE PATH TO DATA FRAME---- participants.the.one.that.works.csv
   |
   |-----src
   |       |
   |       |------__init__.py
   |       |
   |       |------packages
   |                  |
   |                  |-----------__init__.py
   |                  |-----------tools
   |                  |            |
   |                  |            |-------__init__.py
   |                  |            |-------new_data_frames.py    
   |                  |            |-------exploration.py          
   |                  |            |-------analysis.py
   |      
   |-------tests
   |       |
   |       |------__init__.py
   |       |
   |       |------tools
   |                |
   |                |-------__init__.py
                    |-------tests_exploration.py
   |                |-------tests_anlysis.py
   |                |-------test_newdataframe.py
   |
   |
   |-------README.md
   |
   |-------python_gadol.code-workspace
   |
   |-------pyproject.toml
   |
   |-------__main__.py (main script, used to run the whole code smoothly)
   |
   |-------git_ignore.git.ignore
## 6. FUNCTIONS, DATA AND EXPLANATION
   In this section you can find all the functions used in this project and in which modules to find them:
   For example..
       module_name.py:
           function1 -
               explanantion
           function2 -
               explanantion
           function3 -
               explanantion
           etc....








       new_data_frames.py:
       This file is made to clean, manage and extracting the relevant data we will need in order to answer our
       research question.




           def remove_outliers -
             This function replaces outliers in numeric columns of the dataset with NaN. Outliers are identified as values outside 1.5 times the Interquartile Range (IQR).
           def meaning_the_sessions -
           This function computes mean values for groups of columns representing different sessions ( _session1, _session2,_session3) and creates new mean columns. It excludes invalid or completely missing columns.
           def separating_genders -
          This function separates the dataset into two subsets based on the 'sex' column: one for males and one for females. It ensures the 'sex' column is properly formatted before filtering








       explorations.py:
       This file aim is to explore our data and visualize it in a convinient way to the user to better understand the data the user will face in this prohect and understanding the parameters we will use to answer our research question.
           def plot_side_by_side_bar This function takes female_data, male_data, and united_data DataFrames containing mind-wandering dimension scores. extracts, sorts, and computes the mean and standard deviation of these scores, and visualizes them in a side-by-side bar chart. The output is a bar plot comparing mean scores for each dimension across the three groups with error bars indicating standard deviations.
         
           def plot_intrusive_thoughts -
               this show a plot of the data from the instrusive thoughts questionaires
            def compare_sessions_grouped -
                Analyzes and visualizes grouped session data by comparing session means, max-min differences, and overall means for common base metrics.
           def plot_correlation_matrix -
               a heatmap to show correlation between multiple of pairs




       anlysis.py:
           def linear_regression_trial -
               This function performs linear regression to analyze predictors (non-"Mini_Item" columns) against response variables ("Mini_Item" columns) in a dataset. It fills missing values with column means, fits OLS models for each response variable, and plots p-values to highlight significant predictors. The function returns a dictionary of fitted models for further analysis.
           def plot_signi -
               
                This function visualizes relationships between significant predictors (with p-values ≤ 0.05) and response variables from a regression model. For each valid predictor-response pair, it computes the p-value, correlation, regression line, and
                𝑅2, then creates a scatter plot with a regression line and a detailed legend showing these statistics.
           def linear_regression_with_sex_interactions -
               This function fits linear regression models with interaction terms for a binary 'sex' variable, analyzing gender-specific effects. It processes the dataset by handling missing values, encoding 'sex', and creating interaction terms. For each target variable, it fits a model and visualizes interaction p-values in bar plots, highlighting significant predictors. The function returns fitted models and the processed dataset.


           def plot_signi_bysex -
               This function visualizes significant interaction terms (_bysex) from regression models, plotting male and female data separately for each state variable. It calculates correlations, regression lines, and interaction p-values, displaying scatter plots with regression lines for both genders. Titles, labels, and legends highlight the key statistics
## 7. USES
   Just run the main.py file and you will see all data, exploration and analyzes in plots and outputs



















