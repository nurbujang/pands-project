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
print(df.head(5))

# check for missing values
df.info() # returns RangeIndex: 150 entries, 0 to 149, so no missing values
print(df.info())
df.isna().sum() # get number of missing values in each column
print(df.isna().sum())
# count_nan = df.isna().sum().sum() ##############################################
# print ('Count of NaN: ' + str(count_nan)) # get total count of NaN

# 1. Summary into text file, containing basic statistical analysis
df.describe() # to get basic statistical analysis data
print (df.describe())
text_file = open("summary.txt", "wt") # to write the string into a text file
n = text_file.write(df.describe().to_string()) # export into text file called summary, convert pandas dataframe to string
text_file.close()

# 2. Histogram into png file





# # create normal distribution curve, mean=loc, standard deviation=scale
# normData=np.random.normal(loc = 5, scale = 2, size = 1000)
# plt.hist(normData, color='royalblue', edgecolor='#6495ED', linewidth=1)
# # add grid on y-axis
# plt.grid(axis = 'y', color = '#8FBC8F', linestyle = '--', linewidth = 0.3)


# # 3. Scatter plot



# # show plot
# plt.show()

# 4. Other analysis
