
# Visualization Libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

# ML Libraries
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation Metrics
!pip install yellowbrick
from yellowbrick.classifier import ClassificationReport
from sklearn import metrics

import os
import pandas as pd

# Set the directory path to the parent folder containing monthly folders
directory = 'C:/Users/Chhavi/Desktop/DATA_MINING_HW/MERSEYSIDE_ONLY_CRIME'

# Create an empty list to store dataframes for each monthly dataset
dfs = []

# Loop through each monthly folder and concatenate its corresponding dataset
for folder in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, folder)):
        if len(folder) == 7 and '-' in folder:
            year, month = folder.split('-')
            file_path = os.path.join(directory, folder, f'{folder}-merseyside-street.csv')
            if os.path.exists(file_path):
                print(f'Reading in file: {file_path}')
                df = pd.read_csv(file_path)
                df['Year'] = year
                df['Month'] = month
                dfs.append(df)
            else:
                print(f'File not found: {file_path}')
                
# Concatenate all monthly datasets into a single dataframe
if len(dfs) > 0:
    df = pd.concat(dfs, ignore_index=True)
    print('Dataframes concatenated successfully!')
else:
    print('No dataframes to concatenate!')


print(df.head())


df.to_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/merged_crime_data.csv', index=False)




df.info()

import pandas as pd

# Load the merged_crime_data dataset
merged_crime_data = pd.read_csv('C:/New Volume D/DATA MINING/merged_crime_data.csv')

# Drop rows where Crime ID is null
merged_crime_data = merged_crime_data.dropna(subset=['Crime ID'])

# Save the new dataset to a CSV file
merged_crime_data.to_csv('new_merged_crime_data.csv', index=False)


df = df.dropna()


df.info()



import pandas as pd

# read in your data as a pandas dataframe
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# count the total number of rows
num_rows = df.shape[0]

print(f'The dataframe has {num_rows} rows.')


import pandas as pd

# read in your data as a pandas dataframe
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# print the dataframe to check if it's empty
print(df)

# count the total number of rows
num_rows = df.shape[0]

print(f'The dataframe has {num_rows} rows.')

# check if there are enough rows to sample
if len(df) >= 100:
    df = df.dropna().sample(n=100)
else:
    print("Not enough rows to sample from.")




import pandas as pd

# read in your data as a pandas dataframe
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# count the total number of rows
num_rows = df.shape[0]

# set the desired sample size
sample_size = 10000

# check if there are enough rows to sample
if num_rows >= sample_size:
    # randomly sample rows from the dataframe
    df = df.sample(n=sample_size, replace=False)
else:
    print("Not enough rows to sample from.")




import numpy as np

# Create a population of integers from 1 to 10
population = np.arange(1, 11)

# Take a random sample of 5 integers from the population
sample = np.random.choice(a=population, size=5, replace=False)

print("Population:", population)
print("Sample:", sample)



import pandas as pd

# Assuming that df is a pandas DataFrame object
# Get the number of rows in the DataFrame
num_rows = df.shape[0]

# Sample 100,000 rows from df without replacement if it has enough rows
if num_rows >= 100000:
    sample_df = df.sample(n=100000, replace=False)
else:
    sample_df = df.sample(n=num_rows, replace=False)

# Print the first few rows of the sampled DataFrame
print(sample_df.head())


df['Crime type'] = pd.factorize(df["Crime type"])[0]
df['Last outcome category'] = pd.factorize(df["Last outcome category"])[0]


import seaborn as sns

sns.countplot(x="Crime type", data=df)


sns.countplot(x="Last outcome category", data=df)




print(df["Crime type"].unique())
print(df["Last outcome category"].unique())


print(df["Crime type"].value_counts())
print(df["Last outcome category"].value_counts())


print(df["Crime type"].isnull().sum())
print(df["Last outcome category"].isnull().sum())







import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the DataFrame by crime type and count the number of occurrences of each crime type
crime_counts = df.groupby('Crime type')['Crime ID'].count().sort_values()

# Create a horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(crime_counts.index, crime_counts.values)
plt.title('Amount of Crimes by Crime Type')
plt.xlabel('Amount of Crimes')
plt.ylabel('Crime Type')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Create a horizontal bar plot of the crime types
crime_counts = df['Crime type'].value_counts()
crime_counts.plot(kind='barh')

# Set the title and axis labels
plt.title('Number of Crimes by Type')
plt.xlabel('Count')
plt.ylabel('Crime Type')

# Display the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the DataFrame by crime type and count the number of occurrences of each crime type
crime_counts = df.groupby('Crime type')['Crime ID'].count().sort_values()

# Create a horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(crime_counts.index, crime_counts.values)
plt.title('Amount of Crimes by Crime Type')
plt.xlabel('Amount of Crimes')
plt.ylabel('Crime Type')
plt.show()






import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the DataFrame by crime type and count the number of occurrences of each crime type
crime_counts = df.groupby('Crime type')['Crime ID'].count().sort_values()

# Create a colormap from Matplotlib
cmap = plt.get_cmap('viridis')

# Create a horizontal bar plot with colored bars
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(crime_counts.index, crime_counts.values, color=cmap(np.arange(len(crime_counts))))
ax.set_title('Amount of Crimes by Crime Type')
ax.set_xlabel('Amount of Crimes')
ax.set_ylabel('Crime Type')

# Add a colorbar to the plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(crime_counts)))
sm._A = []
cbar = plt.colorbar(sm)

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Define a dictionary that maps each crime type to a color
color_dict = {
    'Anti-social behaviour': 'tab:blue',
    'Bicycle theft': 'tab:orange',
    'Burglary': 'tab:green',
    'Criminal damage and arson': 'tab:red',
    'Drugs': 'tab:purple',
    'Other crime': 'tab:brown',
    'Other theft': 'tab:pink',
    'Possession of weapons': 'tab:gray',
    'Public order': 'tab:olive',
    'Robbery': 'tab:cyan',
    'Shoplifting': 'tab:gray',
    'Theft from the person': 'tab:pink',
    'Vehicle crime': 'tab:orange',
    'Violence and sexual offences': 'tab:green'
}

# Group the DataFrame by crime type and count the number of occurrences of each crime type
crime_counts = df.groupby('Crime type')['Crime ID'].count().sort_values()

# Create a horizontal bar plot with different colors for each crime type
colors = [color_dict[c] for c in crime_counts.index]
plt.figure(figsize=(10, 8))
plt.barh(crime_counts.index, crime_counts.values, color=colors)
plt.title('Amount of Crimes by Crime Type')
plt.xlabel('Amount of Crimes')
plt.ylabel('Crime Type')
plt.show()




crime_counts = df.groupby('Crime type')['Crime ID'].count().sort_values()


threshold = 500 # set a threshold value
df['New Crime Type'] = np.where(df['Crime type'].isin(crime_counts[crime_counts<threshold].index), 'Others', df['Crime type'])


new_crime_counts = df.groupby('New Crime Type')['Crime ID'].count().sort_values()


colors = plt.cm.Set3(np.linspace(0, 1, len(new_crime_counts)))
plt.figure(figsize=(10, 8))
plt.barh(new_crime_counts.index, new_crime_counts.values, color=colors)
plt.title('Amount of Crimes by Crime Type')
plt.xlabel('Amount of Crimes')
plt.ylabel('Crime Type')
plt.show()


# Sum up the amount of Crime Type happened and select the last 13 classes
all_classes = df.groupby(['Crime type'])['Crime ID'].size().reset_index()
all_classes['Amt'] = all_classes['Crime ID']
all_classes = all_classes.drop(['Crime ID'], axis=1)
all_classes = all_classes.sort_values(['Amt'], ascending=[False])

unwanted_classes = all_classes.tail(13)




# Group by Primary Type and count the number of occurrences
all_classes = df.groupby(['Crime type']).size().reset_index(name='Count')

# Sort by Count in descending order
all_classes = all_classes.sort_values(by='Count', ascending=False)

# Select the unwanted classes to be grouped under label 'OTHERS'
unwanted_classes = all_classes.tail(3)

# Replace unwanted classes with 'OTHERS'
df.loc[df['Crime type'].isin(unwanted_classes['Crime type']), 'Crime type'] = 'OTHERS'

# Plot bar chart to visualize Primary Types
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')
df.groupby(['Crime type']).size().sort_values(ascending=True).plot(kind='barh')
plt.show()






df.info()

Classes = df['Crime type'].unique()


Classes

df['Crime Type'] = pd.factorize(df["Crime type"])[0] 
df['Crime Type'].unique()


'''Heatmap is a graphical representation of data in a matrix form using color-coded cells. 
It helps to visualize the correlation between different variables in a dataset. 
The heatmap is created by assigning colors to each cell in the matrix according 
to the value of that cell. This allows easy visualization of patterns and trends 
in the data. In the context of machine learning, heatmaps are often used to identify 
correlations between features in a dataset, and to identify which features are most 
important for predicting a particular target variable.'''


X_fs = df.drop(['Crime type'], axis=1)
Y_fs = df['Crime type']

#Using Pearson Correlation
plt.figure(figsize=(20,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor['Crime Type'])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.2]
relevant_features


Features = ["Year", "Crime type", "Location"]
print('Updated Features: ', Features)



x = df[['Year', 'Crime type', 'Location']]
y = df['Crime Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=3)

print('Feature Set Used : ', x.columns.tolist())
print('Target Class : ', 'Crime Type')
print('Training Set Size : ', x_train.shape)
print('Test Set Size : ', x_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create feature and target arrays
X = df.drop(['Crime type'], axis=1) # Features (all columns except 'Crime type')
y = df['Crime type'] # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
knn_model = KNeighborsClassifier(n_neighbors=5) # Set the number of neighbors to consider

# Model Training
knn_model.fit(X_train, y_train)

# Prediction
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(df.dtypes)






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# One-hot encode categorical variables
ohe = OneHotEncoder()
X_cat = ohe.fit_transform(df[['Crime type', 'Location']]).toarray()
X_num = df[['Longitude', 'Latitude']].values
X = pd.concat([pd.DataFrame(X_cat), pd.DataFrame(X_num)], axis=1)
y = df['Crime type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = knn.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')











import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the data by 'Crime type'
df_violence_sexual = df[df['Crime type'] == 'Violence and sexual offences']
#df_violence_sexual = df[df['Crime type'] == 'violence & sexual offences'].copy()


# Label encode the 'Location' column
le = LabelEncoder()
df_violence_sexual['Location'] = le.fit_transform(df_violence_sexual['Location'])

# Select relevant features for prediction
X = df_violence_sexual[['Location', 'Longitude', 'Latitude']]
y = df_violence_sexual['Crime type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = knn.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')








'''To evaluate the performance of the KNN classifier on a validation set or perform cross-validation, 
you can use the cross_val_score function from Scikit-Learn's model_selection module. 
This function splits the dataset into k folds and evaluates the model on each fold using the specified scoring metric.

Here's an example of how to use cross_val_score to evaluate the KNN classifier's performance on a 
validation set:This code will split the dataset into 5 folds and evaluate the KNN classifier on each 
fold using accuracy as the scoring metric. The cv_scores variable will contain the accuracy scores for 
each fold, and the cv_scores.mean() method call will output the average accuracy across all folds.

'''
from sklearn.model_selection import cross_val_score

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Select relevant features for prediction for 'Violence and sexual offences'
df_violence_sexual = df[df['Crime type'] == 'Violence and sexual offences']
X = df_violence_sexual[['Location', 'Longitude', 'Latitude']]
y = df_violence_sexual['Crime type']

# Label encode the 'Location' column
le = LabelEncoder()
X['Location'] = le.fit_transform(X['Location'])

# Train the KNN classifier using the entire dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Evaluate the performance of the KNN classifier using 5-fold cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Average accuracy: {cv_scores.mean():.2f}')


'''If the cross-validation scores are all 1.0, then it is likely that there is an issue with the data or the evaluation. It is possible that the dataset only contains a single class, which would result in perfect accuracy for any classifier. It could also be the case that the evaluation metric is not appropriate for the problem, or that the data has been overfit in some way.

Here are a few things you can try to diagnose the issue:'''
'''

    Check that the dataset contains more than one class. You can do this by printing out the unique values in the target variable, y, and ensuring that there is more than one value.

    Try using a different evaluation metric to cross-validate the model. For example, you could use precision, recall, or F1 score instead of accuracy. This may reveal that the model is not actually performing as well as it seems.

    Check that the model is not overfitting to the training data. You can do this by setting aside a separate test set and evaluating the model on that. If the model's performance drops significantly on the test set compared to the training set, then it may be overfitting.

    Try using a different classification algorithm to see if the same issue occurs. If the issue persists, then it may be a problem with the data or the evaluation, rather than the model itself.

'''



import pandas as pd

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Print out the unique values in the 'Crime type' column
unique_crime_types = df['Crime type'].unique()
print(f'Unique crime types: {unique_crime_types}')

# Check if there is more than one unique value
if len(unique_crime_types) > 1:
    print('Dataset contains more than one class')
else:
    print('Dataset contains only one class')


from sklearn.metrics import f1_score

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Select relevant features for prediction for 'Violence and sexual offences'
df_violence_sexual = df[df['Crime type'] == 'Violence and sexual offences']
X = df_violence_sexual[['Location', 'Longitude', 'Latitude']]
y = df_violence_sexual['Crime type']

# Label encode the 'Location' column
le = LabelEncoder()
X['Location'] = le.fit_transform(X['Location'])

# Train the KNN classifier using the entire dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Evaluate the performance of the KNN classifier using 5-fold cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5, scoring='f1_macro')
print(f'Cross-validation F1 scores: {cv_scores}')
print(f'Average F1 score: {cv_scores.mean():.2f}')




'''To check for overfitting, you can evaluate the performance of the KNN classifier on both the training set and the test set.
If the accuracy on the test set is significantly lower than the accuracy on the training set, it may indicate overfitting.

To use a different classification algorithm, you can replace the KNN classifier with a different classifier from scikit-learn.
For example, you could try using  random forests.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the data by 'Crime type'
df_violence_sexual = df[df['Crime type'] == 'Violence and sexual offences']

# Label encode the 'Location' column
le = LabelEncoder()
df_violence_sexual['Location'] = le.fit_transform(df_violence_sexual['Location'])

# Select relevant features for prediction
X = df_violence_sexual[['Location', 'Longitude', 'Latitude']]
y = df_violence_sexual['Crime type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rfc.predict(X_test)

# Evaluate the accuracy of the classifier on the training set
train_accuracy = rfc.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Evaluate the accuracy of the classifier on the testing set
test_accuracy = rfc.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')






'''decision - trees'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/Chhavi/Desktop/DATA_MINING_HW/new_merged_crime_data.csv')

# Group the data by 'Crime type'
df_violence_sexual = df[df['Crime type'] == 'Violence and sexual offences']

# Label encode the 'Location' column
le = LabelEncoder()
df_violence_sexual['Location'] = le.fit_transform(df_violence_sexual['Location'])

# Select relevant features for prediction
X = df_violence_sexual[['Location', 'Longitude', 'Latitude']]
y = df_violence_sexual['Crime type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dtc.predict(X_test)

# Evaluate the accuracy of the classifier on the training set
train_accuracy = dtc.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Evaluate the accuracy of the classifier on the testing set
test_accuracy = dtc.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')





