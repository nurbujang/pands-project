# analysis.py
# Pands project for Programming and Scripting
# Author: Nur Bujang
# To write a program called analysis.py that:
# 1. Outputs a summary of each variable to a single text file,
# 2. Saves a histogram of each variable to png files,
# 3. Outputs a scatter plot of each pair of variables.
# 4. Perform any other analysis I think is appropriate

# General Process: Load data, Analyze/visualize dataset, Model training, Model Evaluation, Model Testing, 

# save iris.data file from UCI website and upload into pands-project folder

# import modules
import pandas as pd # for data loading from other sources and processing
import numpy as np # for computational operations
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
#import matplotlib.patches as mpatches # to customize legend
#import matplotlib.lines as mlines # to customize legend
#from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm


from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algorithm


# Load data and add column header
columns = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Iris species'] # define column headers
df=pd.read_csv('iris.data', names=columns) # read the csv file and assign each column name
#print(df.head(5)) # print out first 5 lines

# # check for missing values
# df.info() # returns RangeIndex: 150 entries, 0 to 149, so no missing values
# #print(df.info()) # print out data info

# df.isna().sum() # get number of missing values in each column. df.isnull().sum() can also be used
# #print(df.isna().sum()) # print out number of missing values

# df.drop_duplicates() # remove duplicates
# print(df.drop_duplicates()) # print out remove duplicates
# # output: 150 lines remain, no duplicates exist

# df.value_counts("Iris species") # how many lines for each species
# print(df.value_counts("Iris species")) # print how many lines for each species
# # output: 50 lines for each species

# # 1. Summary into text file, containing basic statistical analysis
# df.describe() # to get basic statistical analysis data
# #print (df.describe()) # print out description of data

# text_file = open("summary.txt", "wt") # to write the string into a text file
# n = text_file.write(df.describe().to_string()) # export into text file called summary, convert pandas dataframe to string
# text_file.close() # always close file

# 2. Histogram of each variable into png file
# Visualizing 4 histograms of each column is not very informative, so overlapping histogram is chosen
# sns.set(style="whitegrid") # set background
# fig,axs = plt.subplots(2,2, figsize = (7,8)) # set grid position of subplots (2 down, 2 across), set size of whole figure (7 width, 8 height)
# fig.suptitle('Petal and sepal dimensions of three Iris species', color ='#191970', fontweight='bold') # customize figure's main title
# sns.histplot(data=df, x="Sepal length (cm)", kde=True, color="olive", ax=axs[0, 0]) # Kernel density estimation (KDE) smooths the replicates with a Gaussian kernel
# sns.histplot(data=df, x="Sepal width (cm)", kde=True, color="green", ax=axs[0, 1]) # ax is the coordinate of each subplot on the figure
# sns.histplot(data=df, x="Petal length (cm)", kde=True, color="blue", ax=axs[1, 0])
# sns.histplot(data=df, x="Petal width (cm)", kde=True, color="purple", ax=axs[1, 1])
# fig.tight_layout() # to fit all subplots into one figure nicely automatically
# plt.savefig('iris.png') # save output into png file
# plt.show() # show plot

# # 3. Scatter plot of each pair of variables
# # df contains 3 classes (setosa, virginicus, versicolor) and 50 replicates each
# # within that, the variables are Petal length, Petal width, Sepal length and Sepal width

# # Perform a Scatter Plot matrix (aka Pair Plot) to see the relationship between a pair of variables within a combination of multiple variables
# sns.pairplot(df, hue='Iris species', markers=["o", "s", "D"], palette='brg', kind='reg', plot_kws={'line_kws':{'color':'blue'}})
# # Where:
# # kind='reg' applies a linear regression line to identify the relationship within the scatter plot
# # to visualize the whole dataset, using 'Iris species' variable to assign different color to different species
# # hue distinguishes different colors, palette is set to brg palette
# # marker o is circle, s is square, D is diamond
# plt.show() # show plot
# # Results:
# # I. setosa is distinctly different and forms a separate cluster from I. virginica and I. versicolor, which shows some pairwise relationship between these two
# # The petal length and width of I. setosa have much narrower distribution compared to the other two species. 
# # there are overlaps in sepal length and width of all three species.

# # 4. Other analysis, Exploratory data analysis (visual techniques to detect outliers, trends/pattern)
# # 4a. Perform Pearson correlation analysis to determine the degree of linear relationship between 2 continuous variables
# # correlation efficient closer to 1 indicates a strong +ve relationship, closer to -1 indicates a strong -ve relationship
# corr = df.corr(method="pearson") # 
# bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(bool) # eliminate upper triangle for better readibility
# corr = corr.where(bool_upper_matrix)
# print(corr)
# # Results:
# # High positive correlation between Petal length & Petal width, Petal length & Sepal length and Sepal length and petal width

# # 4.2 A Correlation matrix to easily visualize the parameters which best correlate with each other
# fig, ax = plt.subplots(figsize=(9,6)) # set size of whole figure (11 width, 6 height)
# fig.suptitle('Correlation matrix of petal and sepal of three Iris species', color ='#191970', fontweight='bold') # customize figure's main title
# # customize heatmap
# # Adjust the axes attribute to “equal” if True so that each cell gets square-shaped.
# # for color bar, default is vertical, but to move it to the bottom, just add 'orientation': 'horizontal' to cbar argument
# h=sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm', square=True, linewidths = 0.1, linecolor='yellow', cbar_kws={'label': 'range', 'shrink': 0.9}) 
# h.set_xticklabels(h.get_xticklabels(), rotation = 0, fontsize = 10) # This sets the xticks "upright" as opposed to sideways in any figure size
# h.set_yticklabels(h.get_yticklabels(),rotation = 0, fontsize = 10) # This sets the yticks "upright" in any figure size
# plt.show() # show plot
# # results show a strong positive correlation between Petal length & Petal width, Petal length & Sepal length, Sepal length & Petal width

# # 4.3 If I group by species:
# df.groupby("Iris species").corr(method="pearson")
# print (df.groupby("Iris species").corr(method="pearson"))
# # Results:
# # Iris setosa: high correlation between Sepal length & Sepal width
# # Iris versicolor: strong correlation between Petal length & Petal width, Petal length & Sepal length
# # Iris virginica: high correlation between Petal length & Sepal length

# # 4.4 Box plot
def graph(y): # define graph of Iris species as y axis
    sns.boxplot(x="Iris species", y=y, data=df)
    sns.stripplot(x="Iris species", y=y, data=df, jitter=True, edgecolor="gray", alpha=0.35)# add stripplot/jitter plot, set transparency (alpha)
plt.figure(figsize=(10,10))
plt.subplot(221) # grid position top left (2 rows, 2 columns, first top)
graph('Sepal length (cm)')
plt.subplot(222) # grid position top right (2 rows, 2 columns, second top)
graph('Sepal width (cm)')
plt.subplot(223) # grid position bottom left (2 rows, 2 columns, first bottom)
graph('Petal length (cm)')
plt.subplot(224) # grid position bottom right (2 rows, 2 columns, second bottom)
graph('Petal width (cm)')
plt.show() # show plot
# Results:
# Iris setosa has the least distributed and smallest petal size
# Iris virginica has the biggest petal size, and Iris versicolor's petal size is between Iris setosa and virginica
# Sepal size may not be a good variable to differentiate species

# Decision Tree classification

# split the data into training (80%) and testing (20%) to detect overfitting (model learned the training data very well but fails on testing)
from sklearn.model_selection import train_test_split # to split the dataset into training and testing
X = df.iloc[:,:2] # store the first two columns (Sepal length and Sepal width) in an array X 
y = df.iloc[:,4] # store the target variable as label into an array y
print(X.shape, y.shape) # display number of rows and columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # split the dataset into training (80%) and testing (20%)
# random_state is the seed of randomness to help reproduce the same results everytime
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # display the shape and label of training and testing set







# # kNN calculates the distance between the data points and predict the correct species class for the datapoint
# # k = number of neighbors/points closest to the test data
# from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
# # Create KNN Classifier
# kNN = KNeighborsClassifier(n_neighbors = 10)
# # Train the model using the training sets
# kNN.fit(X_train, y_train)
# # Predict the response for test dataset
# y_pred = kNN.predict(X_test)
# # evaluate model accuracy
# print("Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

# # Confusion matrix to describe model performance on the test data when we already know the true values/labels
# from sklearn.metrics import confusion_matrix # to describe performance of model on the test data
# # Call a method predict by using an object classifier 'cls_svm'
# y_predict = kNN.predict(X_test)
# # Calculate cm by calling a method named as 'confusion_matrix'
# cm = confusion_matrix(y_test, y_predict)
# # # Call a method heatmap() to plot confusion matrix
# sns.heatmap(cm, annot=True, linewidths=0.1, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
# plt.title("Confusion Matrix",fontsize=20)
# plt.show() # show plot

# #classification report
# from sklearn.metrics import classification_report

# # Display the classification report
# print(classification_report(y_test, y_predict))

# # hyperparameters

# # kNN Classification
# from matplotlib.colors import ListedColormap
# from sklearn import neighbors, datasets
# from sklearn.inspection import DecisionBoundaryDisplay

# n_neighbors = 15

# # import some data to play with
# iris = datasets.load_iris()

# # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
# X = iris.data[:, :2]
# y = iris.target

# # Create color maps
# cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
# cmap_bold = ["darkorange", "c", "darkblue"]

# for weights in ["uniform", "distance"]:
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     clf.fit(X, y)

#     _, ax = plt.subplots()
#     DecisionBoundaryDisplay.from_estimator(
#         clf,
#         X,
#         cmap=cmap_light,
#         ax=ax,
#         response_method="predict",
#         plot_method="pcolormesh",
#         xlabel=iris.feature_names[0],
#         ylabel=iris.feature_names[1],
#         shading="auto",
#     )

#     # Plot also the training points
#     sns.scatterplot(
#         x=X[:, 0],
#         y=X[:, 1],
#         hue=iris.target_names[y],
#         palette=cmap_bold,
#         alpha=1.0,
#         edgecolor="black",
#     )
#     plt.title(
#         "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
#     )

# plt.show()