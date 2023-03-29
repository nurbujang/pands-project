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
import matplotlib.patches as mpatches # for data visualization
import matplotlib.lines as mlines # for data visualization
import seaborn as sns # for data visualization
import matplotlib.patches as mpatches # to customize legend
import matplotlib.lines as mlines # to customize legend

# Load data and add column header
columns = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Iris species'] # define column headers
df=pd.read_csv('iris.data', names=columns) # read the csv file and assign each column name
#print(df.head(5)) # print out first 5 lines

# check for missing values
df.info() # returns RangeIndex: 150 entries, 0 to 149, so no missing values
#print(df.info()) # print out data info

df.isna().sum() # get number of missing values in each column
#print(df.isna().sum()) # print out number of missing values

# 1. Summary into text file, containing basic statistical analysis
df.describe() # to get basic statistical analysis data
#print (df.describe()) # print out description of data

text_file = open("summary.txt", "wt") # to write the string into a text file
n = text_file.write(df.describe().to_string()) # export into text file called summary, convert pandas dataframe to string
text_file.close() # always close file

# 2. Histogram of each variable into png file
# Visualizing 4 histograms of each column is not very informative, so overlapping histogram is chosen
sns.set(style="whitegrid") # set background
fig,axs = plt.subplots(2,2, figsize = (7,8)) # set arrangement of subplots (2 down, 2 across), set size of whole figure (7 width, 8 height)
fig.suptitle('Petal and sepal dimensions of three Iris species', color ='#191970', fontweight='bold') # customize figure's main title
sns.histplot(data=df, x="Sepal length (cm)", kde=True, color="olive", ax=axs[0, 0]) # Kernel density estimation (KDE) smooths the replicates with a Gaussian kernel
sns.histplot(data=df, x="Sepal width (cm)", kde=True, color="green", ax=axs[0, 1]) # ax is the coordinate of each subplot on the figure
sns.histplot(data=df, x="Petal length (cm)", kde=True, color="blue", ax=axs[1, 0])
sns.histplot(data=df, x="Petal width (cm)", kde=True, color="purple", ax=axs[1, 1])
fig.tight_layout() # to fit all subplots into one figure nicely automatically
# plt.savefig('iris.png') # save output into png file
# plt.show() # show plot

# 3. Scatter plot of each pair of variables
# df contains 3 classes (setosa, virginicus, versicolor) and 50 replicates each
# within that, the variables are Petal length, Petal width, Sepal length and Sepal width
# i will use pairplot, and pair up Petal length and width, and Sepal length and width, but confirm with correlation matrix
# correlation matrix will find the parameters which best correlate with each other
# correlation value ranges from -1 to 1, and if 2 variables are highly correlated, we can neglect one
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,4)) # set size of whole figure (4 width, 8 height)
fig.suptitle('Correlation matrix of petal and sepal of three Iris species', color ='#191970', fontweight='bold') # customize figure's main title
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm', square=True, linewidths = .1, linecolor='yellow', cbar_kws={'label': 'range', 'orientation': 'vertical'})
plt.show() # show plot


# 
# plt.show()

# 4. Other analysis
