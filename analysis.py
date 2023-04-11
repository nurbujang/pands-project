"""
Pands project for Programming and Scripting
analysis.py
Author: Nur Bujang

To write a program called analysis.py that:
    1. Outputs a summary of each variable to a single text file,
    2. Saves a histogram of each variable to png files,
    3. Outputs a scatter plot of each pair of variables.
    4. Perform any other analysis I think is appropriate

Save iris.data file from UCI website and upload into pands-project folder
General Process: Load data, Analyze/visualize dataset, Model training, Model Evaluation, Model Testing, 
"""

"""
IMPORT MODULES
"""

import numpy as np # for computational operations
import pandas as pd # for data loading from other sources and processing
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
from sklearn.model_selection import train_test_split # to split data 
from sklearn.svm import SVC # import Support Vector Classification from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""
Load data and add column header
"""
# create a list of column names 
columns = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Iris species']
# create a pandas dataframe object called df, read the csv file and assign a name to each column
df = pd.read_csv('iris.data', names=columns) # df contains iris.data and add column names to the dataframe
print(df.head(2))  # print out first 2 lines to see if the column names were added properly
# Output: Column names were added properly

'''
Quick lookover of the dataset, check for missing values, duplicates
'''
# Quick lookover of the dataset to see the number of unique values
df.value_counts("Iris species")  # how many lines for each species
print(df.value_counts("Iris species"))  # print how many lines for each species
# Output: 50 lines for each species

# basic info about the dataset, column numbers, data types, non-null values
# can also be used to see of there are missing values from the number of non-null values
df.info()  # basic information about the dataframe
print(df.info()) # print out data info
# Output: RangeIndex: 150 entries, 0 to 149

# get number of missing values in each column
df.isnull().sum()
# print (df.isnull().sum()) # print out number of missing values
# OR
# df.isna().sum()
# print(df.isna().sum()) # print out number of missing values

# df.drop_duplicates() # remove duplicates
print(df.drop_duplicates().shape)  # print out remove duplicates
# output: 150 lines remain, no duplicates exist --x wrong code on line 51

'''
Question 1. Summary into text file, containing basic statistical analysis
'''
df.describe()  # to get basic statistical analysis data
# print (df.describe()) # print out description of data in terminal output, but I don't want this

# So, export data into a .txt file called summary.txt
# use the built-in open() function
text_file = open("summary.txt", "wt") # open a new file called summary, mode is w, and t is text mode

# export into text file called summary, convert pandas dataframe to string
n = text_file.write(df.describe().to_string()) # write the converted df.describe() string into a text file
text_file.close()  # always close file when done

"""
3. DATA VISUALISATION: HISTOGRAM

add comments
"""


# 2. Histogram of each variable into png file
sns.set(style="whitegrid")  # set background
# set grid position of subplots (2 down, 2 across), set size of whole figure (7 width, 8 height)
fig, axs = plt.subplots(2, 2, figsize=(7, 8))
fig.suptitle('Petal and sepal dimensions of three Iris species',
             color='#191970', fontweight='bold')  # customize figure's main title
# Kernel density estimation (KDE) smooths the replicates with a Gaussian kernel
sns.histplot(data=df, x="Sepal length (cm)",
             kde=True, color="olive", ax=axs[0, 0])
sns.histplot(data=df, x="Sepal width (cm)", kde=True, color="green",
             ax=axs[0, 1])  # ax is the coordinate of each subplot on the figure
sns.histplot(data=df, x="Petal length (cm)",
             kde=True, color="blue", ax=axs[1, 0])
sns.histplot(data=df, x="Petal width (cm)",
             kde=True, color="purple", ax=axs[1, 1])
fig.tight_layout()  # to fit all subplots into one figure nicely automatically
plt.savefig('iris.png')  # save output into png file
plt.show()  # show plot

"""
3.2 VISUALISATION: SCATTERPLOT MATRIX
"""
# 3. Scatter plot of each pair of variables
# df contains 3 classes (setosa, virginicus, versicolor) and 50 replicates each
# within that, the variables are Petal length, Petal width, Sepal length and Sepal width
# Perform a Scatter Plot matrix (aka Pair Plot) to see the relationship between a pair of variables within a combination of multiple variables
pp = sns.pairplot(df, hue='Iris species', markers=[
                  "o", "s", "D"], palette='brg', kind='reg', plot_kws={'line_kws': {'color': 'blue'}})
plt.suptitle('Pair Plot for sepal and petal attributes of three Iris species',
             fontweight='bold', size=15)
# Where:
# kind='reg' applies a linear regression line to identify the relationship within the scatter plot
# to visualize the whole dataset, using 'Iris species' variable to assign different color to different species
# hue distinguishes different colors, palette is set to brg palette
# marker o is circle, s is square, D is diamond
# handles = pp._legend_data.values()
# labels = pp._legend_data.keys()
# shifts the pairplot position, 0.92 of the default 0.9 for the top edge and 0.08 of the default 0.1 for the bottom edge
pp.fig.subplots_adjust(top=0.92, bottom=0.08)
plt.show()  # show plot
# Results:
# I. setosa is distinctly different and forms a separate cluster from I. virginica and I. versicolor, which shows some pairwise relationship between these two
# The petal length and width of I. setosa have much narrower distribution compared to the other two species.
# there are overlaps in sepal length and width of all three species.


"""
4. CORRELATION ANALYSIS

...

Other analysis, Exploratory data analysis (visual techniques to detect outliers, trends/pattern)
"""

# 4.1 Perform Pearson correlation analysis to determine the degree of linear relationship between 2 continuous variables
# correlation efficient closer to 1 indicates a strong +ve relationship, closer to -1 indicates a strong -ve relationship
corr = df.corr()  # default is already Pearson Correlation (for linear), but can be changed to Kendall and Spearman for non-parametric. eg: method="Spearman"
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(
    bool)  # eliminate upper triangle for better readibility
# Numpy tril function to extract lower triangle or triu to extract upper triangle
# np.ones returns an array of 1 to create a boolean matrix with the same size as the correlation matrix
# astype converts the upper triangle values to False, while the lower triangle will have the True values
corr = corr.where(bool_upper_matrix) # Pandas where() returns same-sized dataframe, but False is converted to NaN on the upper triangle
print(corr)  # print as terminal output
# Results:
# High positive correlation between Petal length & Petal width, Petal length & Sepal length and Sepal length and petal width

# OR, build a Correlation matrix to visualize the parameters which best correlate with each other easier
# set size of whole figure (9 width, 6 height)
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Correlation matrix of petal and sepal of three Iris species',
             color='#191970', fontweight='bold')  # customize figure's main title
h = sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', square=True, linewidths=0.1,
                linecolor='yellow', cbar_kws={'label': 'range', 'shrink': 0.9})  # customize heatmap
# annot=True: if True display the data value in each cell
# square = True: cell is square-shaped
# default color bar is vertical, but to move it to the bottom, just add 'orientation': 'horizontal' to cbar argument
# name color bar label as range and make the color bar smaller to 0.9 the original size
# This sets the xticks "upright" as opposed to sideways in any figure size, just to read easier
h.set_xticklabels(h.get_xticklabels(), rotation=0, fontsize=10)
# This sets the yticks "upright" in any figure size
h.set_yticklabels(h.get_yticklabels(), rotation=0, fontsize=10)
plt.show()  # show plot
# Results:
# Strong positive correlation between Petal length & Petal width, Petal length & Sepal length, Sepal length & Petal width (same as above)

# 4.2 If I group by species, I will get more insights on which attributes are highly correlated for each species:
df.groupby("Iris species").corr(method="pearson")
# print as terminal output
print(df.groupby("Iris species").corr(method="pearson"))
# Results:
# Iris setosa: high correlation between Sepal length & Sepal width
# Iris versicolor: strong correlation between Petal length & Petal width, Petal length & Sepal length
# Iris virginica: high correlation between Petal length & Sepal length

"""
4.3 Box plot
"""


def graph(y):  # define graph of Iris species as y axis
    sns.boxplot(x="Iris species", y=y, data=df)
    # add stripplot/jitter plot, set transparency (alpha)
    sns.stripplot(x="Iris species", y=y, data=df,
                  jitter=True, edgecolor="gray", alpha=0.35)


plt.figure(figsize=(10, 10))
plt.subplot(221)  # grid position top left (2 rows, 2 columns, first top)
graph('Sepal length (cm)')
plt.subplot(222)  # grid position top right (2 rows, 2 columns, second top)
graph('Sepal width (cm)')
plt.subplot(223)  # grid position bottom left (2 rows, 2 columns, first bottom)
graph('Petal length (cm)')
# grid position bottom right (2 rows, 2 columns, second bottom)
plt.subplot(224)
graph('Petal width (cm)')
plt.suptitle('Box Plot for sepal and petal attributes of three Iris species',
             fontweight='bold', size=15)
plt.show()  # show plot
# Results:
# Iris setosa has the least distributed and smallest petal size
# Iris virginica has the biggest petal size, and Iris versicolor's petal size is between Iris setosa and virginica
# Sepal size may not be a good variable to differentiate species

"""
5. DATA PREPARATION FOR BASIC MACHINE LEARNING

5.1 SPLITTING ....

# SPLITTING THE DATA FOR TRAINING AND TESTING
# First, split the data into training (80%) and testing (20%) to detect overfitting (model learned the training data very well but fails on testing)
# Later, the testing dataset will be used to check the accuracy of the model.
# Most ML examples available use this:
# X = df.iloc[:,:2] # take everything until the second column (columns 0 and 1) and store the first two columns (Sepal length and Sepal width) into attributes (X)
# y = df.iloc[:,4] # store the target variable (Iris species) into labels (y)

"""

X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values
print(X.shape, y.shape)  # display number of rows and columns
# split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# random_state is the seed of randomness to help reproduce the same results everytime
# display the shape and label of training and testing set
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


"""
5.2 kNN ...
# 4.4 Create kNN Classification to plot the species boundaries
# kNN calculates the distance between the data points and predict the correct species class for the datapoint
# where k = number of neighbors/points closest to the test data
# Calculate the accuracy of the model using testing dataset
"""
for i in np.arange(7, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    print("\nk-Nearest Neighbor model accuracy for k = %d accuracy is" % i, knn.score(X_test,
          y_test)*100)  # keep k small because there are only 3 species, to prevent overfitting

"""
# 4.5 Logistic Regression
# to estimate the relationship between 1 dependent variable and 1 or more independent variables
# used for classification and prediction
"""

lr = LogisticRegression(C=0.02)  
# C value is a model hyperparameter, which is a model criteria outside of the model and its value cannot be estimated from the data
# High C value means training data is more reliable (reflects real world data), low C value means training data may not reflect real world data
lr.fit(X_train, y_train)  # train the model
# compare model’s output (y_pred) with target values that we already have (y_test)
y_pred_lr = lr.predict(X_test)
print('\nLogistic Regression model accuracy is',
      accuracy_score(y_test, y_pred_lr) * 100)
print('Logistic Regression model F1 score is',
      f1_score(y_test, y_pred_lr, average='macro'))

# 4.6 Decision Tree Classification
# to build a classification in a form of tree structure with decision nodes and leaf nodes
# branches bifurcate based on Y/N or T/F
# define Decision Tree classifer object
dtclassifier = DecisionTreeClassifier(random_state=42)
dtclassifier.fit(X_train, y_train)  # Train Decision Tree Classifer
# Predict the response for test dataset
y_pred_dt = dtclassifier.predict(X_test)
# Evaluate the model using testing dataset
disp = ConfusionMatrixDisplay.from_estimator(dtclassifier, X_test, y_test)
#disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.grid(False)
plt.suptitle('Decision Tree Confusion Matrix for sepal and petal attributes of three Iris species',
             fontweight='bold', size=10)
print('\nDecision Tree Classification model accuracy is',
      accuracy_score(y_test, y_pred_dt)*100)  # print out accuracy score
print('Decision Tree Classification model F1 score is',
      f1_score(y_test, y_pred_dt, average='macro'))
plt.show()

# 4.7 Support Vector Machine
# for regression and classification
# to create the best boundary to separate data into classes by creating a line with the most margin from the data point
svclassifier = SVC()
svclassifier.fit(X_train, y_train)
y_pred_svc = svclassifier.predict(X_test)  # Predict from the test dataset
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
# Accuracy score using testing dataset
print('\nSupport Vector Machine model accuracy is', accuracy_score(
    y_test, y_pred_svc)*100)  # print out accuracy score
print('Support Vector Machine model F1 score is',
      f1_score(y_test, y_pred_svc, average='macro'))

# 4.8 Random Forest
# to create a cluster of decision trees
# each bunch is trained on random subsets from training group (drawn with replacement) and features (drawn without replacement)
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
rf.fit(X_train, y_train)  # train the model
# compare model’s output (y_pred) with the target values that we already have (y_test)
y_pred_rf = rf.predict(X_test)
# Random Forest visualization using confusion matrix
disp= ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.suptitle('Random Forest Confusion Matrix for sepal and petal attributes of three Iris species',
             fontweight='bold', size=10)
plt.grid(False)
plt.show()
# Accuracy score using testing dataset
print('\nRandom Forest model accuracy score is',
      accuracy_score(y_test, y_pred_rf)*100)
print('Random Forest model F1 score is', f1_score(
    y_test, y_pred_rf, average='macro'))
