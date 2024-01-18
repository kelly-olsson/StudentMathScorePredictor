
import pandas  as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PATH     = "./visualizations/"
CSV_DATA = "StudentsPerformance.csv"

# Get the student data
df = pd.read_csv(PATH + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df)

# print the number of unique values in each column
print(df.nunique())

def convert_string_columns_to_numeric(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# Convert string columns to numerics
df = convert_string_columns_to_numeric(df)

print(df.describe().T)

print(df)

cols = ['math score'] + [col for col in df.columns if col != 'math score']
data = df[cols]

# Calculate the correlation matrix
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', square=True)

# Set the title
plt.title("Correlation Heatmap with Student Performance Dataset")

# Show all text
plt.tight_layout()

# Display the plot
plt.show()

# boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(ax=axes[0], y='math score', data=df, palette='Set2')
axes[0].set_title('Math Score Outliers', fontsize=16)

sns.boxplot(ax=axes[1], y='reading score', data=df, palette='Set2')
axes[1].set_title('Reading Score Outliers', fontsize=16)

sns.boxplot(ax=axes[2], y='writing score', data=df, palette='Set2')
axes[2].set_title('Writing Score Outliers', fontsize=16)

plt.show()

# data distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, column in enumerate(df.columns):
    sns.histplot(ax=axes[i//4, i%4], data=df, x=column, kde=True, bins=20)
    axes[i//4, i%4].set_title(f'Distribution of {column.capitalize()}', fontsize=14)

plt.show()

# Pair plot
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Features with Math Scores as Target', fontsize=16, y=1.02)
plt.show()

# Violin plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.violinplot(x='test preparation course', y='math score', data=data, ax=axes[0])
axes[0].set_title('Math Score by Test Preparation Course')
sns.violinplot(x=pd.cut(data['math score'], bins=5), y='writing score', data=data, ax=axes[1])
axes[1].set_title('Writing Score by Math Score')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
sns.violinplot(x=pd.cut(data['math score'], bins=5), y='reading score', data=data, ax=axes[2])
axes[2].set_title('Reading Score by Math Score')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

# Display the plots
plt.tight_layout()
plt.show()

# Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='test preparation course', y='math score', data=df, palette='coolwarm', errorbar=None)
plt.title('Bar Plot of Test Preparation Course vs Math Scores', fontsize=16)
plt.show()

# Scatter plot
scores_data = df[['math score', 'reading score', 'writing score']]
sns.pairplot(scores_data)
plt.suptitle("Scatter Plot Matrix for Math, Reading, and Writing Scores", y=1.02)
plt.show()
