import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,confusion_matrix,classification_report

# Here is the Data from
url = "https://raw.githubusercontent.com/mdogy/dataForEng1999/refs/heads/master/family_heights.csv"

df = pd.read_csv(url)

# Setting the Data frame to use for X and Y 
numerical_df = df[['father', 'mother', 'midparentHeight','childHeight','childNum','sex']]
classification_df = df[['father', 'mother', 'midparentHeight','childHeight','children','sex']]

# Transform category to numeric
numerical_df['sex_num'] = numerical_df['sex'].map({'male': 1, 'female': 0})





