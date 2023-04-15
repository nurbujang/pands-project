'''
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
'''

'''
IMPORT MODULES
'''
import numpy as np # for computational operations
import pandas as pd # for data loading from other sources and processing
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
from sklearn.model_selection import train_test_split # to split data 
from sklearn.neighbors import KNeighborsClassifier # import kNN Classifier from sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score # sklearn module to print confusion matrix and report
from sklearn.linear_model import LogisticRegression # import Logistic Regression from sklearn
from sklearn.tree import DecisionTreeClassifier # import Decision Tree Classifier from sklearn
from sklearn.svm import SVC # import Support Vector Machine Classification from sklearn
from sklearn.ensemble import RandomForestClassifier # import Random Forest Classifier from sklearn
from sklearn.naive_bayes import GaussianNB # import Gaussian Naïve Bayes from sklearn

'''
Load data and add column header
'''
# create a list of column names 
columns = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Iris species']
# create a pandas dataframe object called df, read the csv file and assign a name to each column
df = pd.read_csv('iris.data', names=columns) # df contains iris.data and add column names to the dataframe
print('\nThe first 2 lines of the dataset\n',df.head(2))  # print out first 2 lines to see if the column names were added properly
# Output: Column names were added properly

'''
PRE-PROCESSING - Quick lookover on the dataset, check for missing values, duplicates
'''
# Quick lookover of the dataset to see the number of unique values
df.value_counts("Iris species")  # how many lines for each species
print('\nThe number of rows for each Iris species\n',df.value_counts("Iris species"))  # print how many lines for each species
# Output: 50 lines for each species

# basic info about the dataset, column numbers, data types, non-null values
# can also be used to see of there are missing values from the number of non-null values
df.info()  # print out in terminal output the basic information about the dataframe
# print(df.info()) # commented out because I don't want to print it twice
# Output: 
# It is a dataframe
# 150 rows, 0 to 149, 5 data columns and column names
# number of non-null values (data that is not missing) in each column
# 4 float datatypes and 1 object

# get number of missing values in each column
df.isnull().sum() # sum of null values in the dataset
print ('\nThe number of missing value\n',df.isnull().sum()) # print out number of missing values
# OR
# df.isna().sum() # sum of na values
# print(df.isna().sum()) # print out number of missing values

df.drop_duplicates() # remove duplicates from the dataset
print('\nData shape after duplicate removal is',df.drop_duplicates().shape)  # print out the shape of the dataset after duplicate removal
# output: 147 rows, 5 columns, meaning 3 rows were removed

'''
Question 1. SUMMARY into text file, containing basic statistical analysis
df. describe to get count, mean, standard deviation, min and max values, lower, mid and upper percentile
'''
df.describe()  # to get basic statistical analysis data 
# print (df.describe()) # print out description of data in terminal output, but I do not want this

# So, export data into a .txt file called summary.txt
# use the built-in open() function
text_file = open("summary.txt", "wt") # open a new file called summary, mode is w, and t is text mode

# export into text file called summary, convert pandas dataframe to string
n = text_file.write(df.describe().to_string()) # write the converted df.describe() string into a text file
text_file.close()  # ALWAYS close file when done

'''
Question 2. DATA VISUALISATION: Histogram of each variable into png file
'''
sns.set(style="whitegrid")  # set background
# set grid position of subplots (2 down, 2 across), set size of whole figure (7 width, 8 height)
fig, axs = plt.subplots(2, 2, figsize=(7, 8)) # create a figure containing subplots with multiple axes 
# set subplot arrangement in 2 directions (2D grid) (2 by 2) 
# set figure size (width and height)
sns.histplot(data=df, x="Sepal length (cm)", # x-axis is sepal length
             kde=True, color="olive", ax=axs[0, 0]) # subplot location first row, first column
# Where:
# seaborn histplot was used to plot the histogram
# dataset used was df
# determine x-axis label
# display KDE line : Kernel density estimation (KDE) shows the data using a continuous curve
# creates data 'smoothing' of the density distribution with a Gaussian kernel
# set the bar color for this plot to olive
# ax is the coordinate of each subplot on the figure
sns.histplot(data=df, x="Sepal width (cm)", kde=True, color="green", # use seaborn histplot, x-axis is sepal  width
             ax=axs[0, 1])  # subplot location first row, second column
sns.histplot(data=df, x="Petal length (cm)", # x-axis is petal length
             kde=True, color="blue", ax=axs[1, 0]) # subplot location second row, first column
sns.histplot(data=df, x="Petal width (cm)", # x-axis is petal width
             kde=True, color="purple", ax=axs[1, 1]) # subplot location second row, second column
fig.suptitle('Histogram of petal and sepal dimensions of three Iris species',
             color='#191970', fontweight='bold')  # customize figure's super title, font color and bold
fig.tight_layout()  # to fit all subplots into one figure nicely automatically
plt.savefig('iris.png')  # save output into png file
plt.show()  # show plot

'''
Question 3. DATA VISUALISATION: Scatter plot of each pair of variables
'''
# df contains 3 classes (setosa, virginicus, versicolor) and 50 replicates each
# within that, the variables are Petal length, Petal width, Sepal length and Sepal width
# Perform a Scatter Plot matrix (aka Pair Plot) to see the relationship between a pair of variables within a combination of multiple variables
# to visualize the whole dataset, using 'Iris species' variable to assign different color to different species
pp = sns.pairplot(df, hue='Iris species', markers=[ # instantiate a seaborn pairplot called pp
                  "o", "s", "D"], palette='brg', kind='reg', plot_kws={'line_kws': {'color': 'blue'}})
# Where:
# dataset used was df
# seaborn pairplot was used to plot the Scatter Plot
# The diagonal plots represent each of that column variable's data distribution
# hue colors the plot based on the different Iris species value
# marker o is circle, s is square, D is diamond
# palette is set to brg palette
# kind is a chart type, kind='reg' applies a linear regression line to identify the relationship within the scatter plot
# plot_kws and pass in a dictionary object to customize the regression fit and line to blue
plt.suptitle('Scatter Plot for sepal and petal attributes of three Iris species',
             fontweight='bold', size=15) # customize figure's super title, default black in bold, fontsize 15
pp.fig.subplots_adjust(top=0.92, bottom=0.08)
# fig.subplots_adjust shifts the pairplot position, 0.92 of the default 0.9 for the top edge and 0.08 of the default 0.1 for the bottom edge
# this was done to show the supertitle properly
plt.show()  # show plot
# Results:
# I. setosa is distinctly different and forms a separate cluster from I. virginica and I. versicolor, which shows some pairwise relationship between these two
# The petal length and width of I. setosa have much narrower distribution compared to the other two species.
# there are overlaps in sepal length and width of all three species.

'''
Question 4. OTHER ANALYSIS:
Data visualization to detect outliers, trends/pattern and Basic Machine Learning analysis
'''

'''
4.1 Pearson correlation analysis 
to determine the degree/strength of linear relationship between 2 continuous variables
correlation efficient closer to 1 indicates a strong +ve relationship, closer to -1 indicates a strong -ve relationship
'''
corr = df.corr(method="pearson")  # create/instantiate an object called corr 
# default is already Pearson Correlation (for linear), but can be changed to Kendall and Spearman for non-parametric. eg: method="Spearman"
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(bool)  # eliminate upper triangle for better readibility
# Numpy tril function to extract lower triangle or triu to extract upper triangle
# np.ones returns an array of 1 to create a boolean matrix with the same size as the correlation matrix
# astype converts the upper triangle values to False, while the lower triangle will have the True values
corr = corr.where(bool_upper_matrix) # Pandas where() returns same-sized dataframe, but False is converted to NaN on the upper triangle
print('\nPearson Correlation by attributes\n',corr)  # print as terminal output
# Results:
# High positive correlation between Petal length & Petal width, Petal length & Sepal length and Sepal length and petal width

# OR, 

# build a Correlation matrix to visualize the parameters which best correlate with each other easier
# set size of whole figure (9 width, 6 height)
fig, ax = plt.subplots(figsize=(9, 6)) # create 1 figure and a single axes
fig.suptitle('Correlation matrix for petal and sepal attributes of three Iris species',
             color='#191970', fontweight='bold')  # customize figure's main/super title
hm = sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', square=True, linewidths=0.1, # create a seaborn heatmap called hm
                linecolor='yellow', cbar_kws={'label': 'Range', 'shrink': 0.9})  # customize heatmap
# Where:
# annot=True: if True, the data value in each cell will be displayed
# coolwarm color palette for the heatmap
# square = True: cell is square-shaped
# linewidth is the width of lines separating each cell and the color is yellow
# I passed arguments into the color bar to show Range as label and shrank it to 0.9 times the original size
# default color bar is vertical, but to move it to the bottom, just add 'orientation': 'horizontal' to cbar argument
hm.set_xticklabels(hm.get_xticklabels(), rotation=0, fontsize=10)
# set the label for x-axis to follow each column name
# rotation sets the xticks "upright" as opposed to sideways in any figure size, just to read easier
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize=10)
# set the label for y axis to follow each column name
# rotation sets the yticks "upright" in any figure size
plt.show()  # show plot
# Results:
# Strong positive correlation between Petal length & Petal width, Petal length & Sepal length, Sepal length & Petal width (same as above)

'''
4.2 If I group by species:
to get more insights on which attributes are highly correlated for each Iris species:
'''
df.groupby("Iris species").corr(method="pearson") # Pearson Correlation, but grouped by class
print('\nPearson Correlation by species\n',df.groupby("Iris species").corr(method="pearson")) # print as terminal output
# Results:
# Iris setosa: high correlation between Sepal length & Sepal width
# Iris versicolor: strong correlation between Petal length & Petal width, Petal length & Sepal length
# Iris virginica: high correlation between Petal length & Sepal length

'''
4.3 DATA VISUALIZATION: Box plot
to display data point distribution/spread, skewness, variance and outliers
shows the minimum, first quartile, median, third quartile and maximum
'''
def graph(y):  # define a graph function of y axis
    sns.boxplot(x="Iris species", y=y, data=df) # seaborn boxplot, with Iris species on x-axis
# on x-axis is Iris species, on y-axis is y (attributes) and the data used is the iris dataframe
    sns.stripplot(x="Iris species", y=y, data=df, # added a seaborn stripplot/jitter plot over the boxplot
                  jitter=True, edgecolor="red", alpha=0.35, linewidth=1)
# Where:
# on x-axis is Iris species, on y-axis is y (attributes) and the data used is the iris dataframe
# jitter=True will display the dots (jitter)
# the edgecolor of the jitter/dots is red to make the dots look clearer
# linewidth is width of line around the dots, it has to be set to >0 to be visible
# set transparency (alpha) to 0.35 so it is not too opaque
plt.figure(figsize=(10, 10))
plt.subplot(221)  # grid position top left (2 rows, 2 columns, first top)
graph('Sepal length (cm)') # y-axis is Sepal length (cm)
plt.subplot(222)  # grid position top right (2 rows, 2 columns, second top)
graph('Sepal width (cm)') # y-axis is Sepal width (cm)
plt.subplot(223)  # grid position bottom left (2 rows, 2 columns, first bottom)
graph('Petal length (cm)') # y-axis is Petal length (cm)
plt.subplot(224) # grid position bottom right (2 rows, 2 columns, second bottom)
graph('Petal width (cm)') # y-axis is Petal width (cm)
plt.suptitle('Box Plot for sepal and petal attributes of three Iris species',
             fontweight='bold', size=15) # customize the figure's super title
plt.show()  # show plot
# Results:
# Iris setosa has the least distributed and smallest petal size
# Iris virginica has the biggest petal size, and Iris versicolor's petal size is between Iris setosa and virginica
# Sepal size may not be a good variable to differentiate species

'''
4.4 DATA VISUALIZATION: Violin Plot
to give more insight into data distribution and density on the y-axis
it contains all data points, unlike the box plot which shows minimum, first quartile, median, third quartile and maximum and error bars
'''
fig, axs = plt.subplots(1, len(columns)-1, figsize=(20,5)) # create a figure containing subplots with multiple axes 
# plot the subplots in 1 row of 4 subplots (stacked in 1 direction only, but still use axs because each has their own axes)
# length -1 means all columns (5) -1 = 4 columns
# set the figure size to 20 width and 5 height
for i in range(0,len(columns)-1): 
    sns.violinplot(x='Iris species', y=df[columns[i]], data=df, ax=axs[i])
    axs[i].set_ylabel(columns[i])
# Where:
# for i in range can be translated to: for item in columns 1 to 4
# seaborn violinplot, x-axis is Iris species, y is each column name, data is iris dataframe
# ax the object to draw the plot into, in this case, the columns
# axs[i].set_ylabel(columns[i]) means each column list i is set as the y-axis label
plt.suptitle('Violin Plot for sepal and petal attributes of three Iris species',
             fontweight='bold', size=15) # customize the figure's super title
plt.show() # show plot

'''
*************DATA PREPARATION FOR BASIC MACHINE LEARNING***************

SPLITTING THE DATA FOR TRAINING AND TESTING
First, split the data into training (80%) and testing (20%) to detect overfitting (=model learned the training data very well but fails on testing)
Later, the testing dataset will be used to check the accuracy of the model.

While most ML examples available uses only 2 variables (sepal length and width only or petal length and width only) to simplify analysis:
X = df.iloc[:,:2] # take everything until the second column (columns 0 and 1) and store the first two columns (Sepal length and Sepal width) into attributes (X)
y = df.iloc[:,4] # store the target variable (Iris species) into labels (y)
I used all 4 variables because the best determinants are still unknown at this point (whether sepal or petal attribute is better than the other).
'''

X = df.iloc[:, :-1].values # everything up until the last column but not including the last column (Iris species) 
y = df.iloc[:, 4].values # = [:, -1],  get all the rows in the last/5th column (target variable (Iris species) into labels (y))
print('\nDataset shape before Train-Test split is',X.shape, y.shape)  
# print out the shape (number of rows and columns) AFTER the data split
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42) # split the dataset into training (80%) and testing (20%)
# test size is 20%, meaning training size is 80%
# random_state is the seed of randomness to help get the same results everytime. It can be any number, really.
print('\nDataset shape after Train-test split is',X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
# print out the shape (number of rows and columns) of test and train data AFTER split in terminal output

'''
4.5 kNN Classifier
used for regression and classification
kNN calculates the distance between the data points and predict the correct species class for the datapoint
where k = number of neighbors/points closest to the test data
'''
for i in np.arange(7, 10): # for i in numpy arange starting at 7 and stopping at 10
# keep k small because there are only 3 species, to prevent overfitting
    knn = KNeighborsClassifier(n_neighbors=i) # instantiate a class and name it knn
    # number of neighbors range from 7-10
    knn.fit(X_train, y_train) # fits the model to the training set
    y_pred_knn = knn.predict(X_test) # Predict the model using the test dataset
    print("\nk-Nearest Neighbor model accuracy for k = %d accuracy is" % i, knn.score(X_test,
          y_test)*100)  # Calculate and print the accuracy of the model using testing dataset ranging (from 7-10)*100 
# %d is a placeholder for a number, %s is for string
# % on its own means the values of each i in 7-10 range are then passed in through a tuple using the % operator
    print("k-Nearest Neighbor F1 score for k = %d F1 score is" % i, f1_score(y_test, y_pred_knn, average='micro'))  
# Calculate and print the F1 score the model using testing dataset ranging (from 7-10)*100 
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive
# Other averaging methods are macro (meaning it is calculated using regular/unweighted mean) and weighted (takes into account each class's support)
# F1 score measures the accuracy of the model, that is how many times it makes a correct prediction in the whole dataset
# F1 combines precision (% of correct predictions) and recall (proportion of correct predictions over total occurences)
      
'''
4.6 Logistic Regression
To estimate the relationship between 1 dependent variable and 1 or more independent variables
Used for classification and prediction analysis
uses Sigmoid function (logistic function) instead of linear function in Linear regression
'''
lr = LogisticRegression(C=0.5, random_state=42)  # instantiate a class and name it lr
# C value is a model hyperparameter, which is a model criteria outside of the model and its value cannot be estimated from the data
# default C is 1.0
# High C value means training data is more reliable (reflects real world data), low C value means training data may not reflect real world data
# I picked 0.5 to be somewhere in the middle
# random_state is the seed of randomness to help get the same results everytime. It can be any number
lr.fit(X_train, y_train)  # train the model
# compare model’s output (y_pred) with target values that we already have (y_test)
y_pred_lr = lr.predict(X_test) # Predict the model using the test dataset
print('\nLogistic Regression model accuracy is',
      accuracy_score(y_test, y_pred_lr) * 100) # print accuracy score * 100
# Model accuracy score is how many times the model makes correct predictions over the total number of predictions
print('Logistic Regression model F1 score is',
      f1_score(y_test, y_pred_lr, average='micro')) # print out F1 score
# F1 score measures the accuracy of the model, that is how many times it makes a correct prediction in the whole dataset
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive
'''
4.7 Decision Tree Classifier
to build a classification or regression 
branches bifurcate based on Y/N or T/F and breaks the dataset smaller everytime
to eventually form a tree structure with decision nodes and leaf nodes
'''
# define Decision Tree classifer object
dtclassifier = DecisionTreeClassifier(random_state=42) # instantiate a class and name it dtclassifier
dtclassifier.fit(X_train, y_train)  # Train Decision Tree Classifer model
# Predict the response for test dataset
y_pred_dt = dtclassifier.predict(X_test) # Predict the model using the test dataset
# Evaluate the model using testing dataset
disp = ConfusionMatrixDisplay.from_estimator(dtclassifier, X_test, y_test)
# Confusion matrix contains Actual values of Positive(1) and Negative(0) on the x-axis and Predicted values of Positive(1) and Negative(0) on the y-axis
# CM for iris contains Actual setosa, versicolor, virginica on the x-axis and Predicted setosa, versicolor, virginica on the y-axis
# True Positive = the actual and predicted value should be the same = 10
# True Negative = 9+0+0+11 = 20
# False Positive = 0+0 (across) = 0
# False Negative = 0+0 (down) = 0
plt.suptitle('Decision Tree Confusion Matrix for sepal and petal attributes of three Iris species',
             fontweight='bold', size=10) # customize super title
plt.grid(False) # eliminate white grid within the confusion matrix
plt.show() # display Confusion Matrix
print('\nDecision Tree Classification model accuracy is',
      accuracy_score(y_test, y_pred_dt)*100)  # print out accuracy score *100
print('Decision Tree Classification model F1 score is',
      f1_score(y_test, y_pred_dt, average='micro')) # print out F1 score
# Precision = True Positive : (True Positive + False Positive) = 10/(10+0+0 (across)) = 1
# Recall = True positive : (True Positive + False Negative) = 10/(10+0+0 (down)) = 1
# F1 score = 2*((precision*recall)/(precision+recall)) = 2(1/2) = 1
# that is, F1 = how many times it makes a correct prediction in the whole dataset
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive


'''
4.8 Support Vector Machine Classifier
for regression and classification
to map data points into a high dimensional space
and then create the best boundary to separate data into classes by creating a hyperplane line with the most margin from the data point
'''
svclassifier = SVC() # instantiate a class and name it svclassifier
svclassifier.fit(X_train, y_train) # train the model
y_pred_svc = svclassifier.predict(X_test)  # Predict the model using the test dataset
print('\nClassification report for Support Vector Classifier\n',classification_report(y_test, y_pred_svc)) # print out classification report in terminal output
# Classification report results:
# Precision = Ratio of True Positive : (True Positive + False Positive. So, setosa (10/10=1), versicolor (9/9=1), virginica (11/11=1)
# Recall = Ratio of True positive : (True Positive + False Negative)
# F1 score = 2*((precision*recall)/(precision+recall)) = 2(1/2) = 1
# Support = number of actual occurences of that class
# Accuracy = correct predictions for all classes / total number of predictions = 10/10 = 1
print('Confusion matrix for Support Vector Classifier\n',confusion_matrix(y_test, y_pred_svc)) # print out Confusion matrix in terminal output
print('\nSupport Vector Classifier model accuracy is', accuracy_score(
    y_test, y_pred_svc)*100)  # print out accuracy score *100
print('Support Vector Classifier model F1 score is',
      f1_score(y_test, y_pred_svc, average='micro')) # print out F1 score
# F1 score measures the accuracy of the model, that is how many times it makes a correct prediction in the whole dataset
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive

'''
4.9 Random Forest Classifier
used to perform both regression and classification
to create a cluster of decision trees containing different sub-features from the features, thus forming a 'forest'
each bunch is trained on random subsets from training group (drawn with replacement and can be used again) and features (drawn without replacement and cannot be reused)
each tree picks the features randomly, making it possible to find which features are more important than others
'''
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1) # instantiate a class and name it rf
rf.fit(X_train, y_train)  # train the model
# compare model’s output (y_pred) with the target values that we already have (y_test)
y_pred_rf = rf.predict(X_test) # Predict the model using the test dataset
# Random Forest visualization using confusion matrix
disp= ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.suptitle('Random Forest Confusion Matrix for sepal and petal attributes of three Iris species',
             fontweight='bold', size=10) # customize super title
plt.grid(False) # eliminate white grid within the confusion matrix
plt.show() # show plot
print('\nRandom Forest model accuracy score is',
      accuracy_score(y_test, y_pred_rf)*100) # print accuracy score * 100
print('Random Forest model F1 score is', f1_score(
    y_test, y_pred_rf, average='micro')) # print out F1 score
# F1 score measures the accuracy of the model, that is how many times it makes a correct prediction in the whole dataset
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive

'''
4.10 Gaussian Naïve Bayes Classifier
'Naive' because it assumes that each variable are independent of each other
it predicts the probability of different species based on different attributes
I used Gaussian because data is continuous, and assumed to be normally-distributed
'''
gaussian = GaussianNB() # instantiate a class and name it gaussian
gaussian.fit(X_train, y_train) # train the model
y_pred_gs = gaussian.predict(X_test) # Predict the model using the test dataset
cm = confusion_matrix(y_test, y_pred_gs) # instantiate confusion matrix
accuracy = accuracy_score(y_test,y_pred_gs)*100 # instantiate accuracy score
# multiply by 100 here because it is too complicated to do so in the print format
f1 = f1_score(y_test,y_pred_gs,average='micro') # instantiate an object and name it f1
# Average method setting is set to micro, meaning the average takes into account the sum of True Positive, False Negative and False Positive
print('\nClassification report for Naive-Bayes Classifier\n',classification_report(y_test, y_pred_gs)) # print Classification Report in terminal output
print('\nConfusion matrix for Naive Bayes\n',cm) # print confusion matrix in terminal output
print('\nNaive-Bayes model accuracy score is %.1f' %accuracy) # .1f is float with 1 decimal point of the accuracy value
print('Naive-Bayes model F1 score is %.3f' %f1) # # print out F1 score, .3f is float with 3 decimal points of the f1 value
# %.1f and %.3f are format specifiers. They begin with %, then followed by a character that represents the data type, which is a float
# F1 score measures the accuracy of the model, that is how many times it makes a correct prediction in the whole dataset
